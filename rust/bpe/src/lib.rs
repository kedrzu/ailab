//! Trening BPE w Ruście: ten sam naiwny algorytm co `src/ailab/bpe.py`.
//!
//! Każda runda liczy wszystkie pary od nowa, bez indeksu „para → słowa”. Zmieniają się tylko
//! język, zapis tokenów jako liczb oraz wątki, żeby dało się zmierzyć zysk z każdej z tych rzeczy.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

type Pair = (u32, u32);

/// Jeden unikalny kawałek tekstu: symbole jako numery tokenów i liczba wystąpień.
struct Word {
    symbols: Vec<u32>,
    count: u64,
}

/// Liczy sąsiadujące pary, ważone licznikiem słowa.
///
/// Każdy wątek buduje własną mapę (`fold`), a potem mapy są łączone parami (`reduce`).
/// Wątki nie dzielą jednej mapy, bo blokowanie jej przy każdym zliczeniu zabiłoby zysk.
fn count_pairs(words: &[Word], reduce_nanos: &AtomicU64, reduce_calls: &AtomicU64) -> HashMap<Pair, u64> {
    // Domyślnie rayon tnie listę na tysiące kawałków, a każdy kawałek to osobna mapa do złączenia.
    // Tu podział jest kosztowny, więc wymuszamy około jednego kawałka na wątek.
    let min_chunk = (words.len() / rayon::current_num_threads()).max(1);
    words
        .par_iter()
        .with_min_len(min_chunk)
        .fold(HashMap::new, |mut counts: HashMap<Pair, u64>, word| {
            for window in word.symbols.windows(2) {
                *counts.entry((window[0], window[1])).or_insert(0) += word.count;
            }
            counts
        })
        .reduce(HashMap::new, |a, b| {
            let started = Instant::now();
            // Dokładamy mniejszą mapę do większej, żeby wykonać jak najmniej wstawień.
            let (mut large, small) = if a.len() >= b.len() { (a, b) } else { (b, a) };
            for (pair, count) in small {
                *large.entry(pair).or_insert(0) += count;
            }
            reduce_nanos.fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
            reduce_calls.fetch_add(1, Ordering::Relaxed);
            large
        })
}

/// Wybiera najczęstszą parę. Remisy rozstrzyga para napisów, tak samo jak w Pythonie:
/// porównanie bajtów UTF-8 daje tę samą kolejność co porównanie punktów kodowych.
fn most_frequent_pair(pairs: &HashMap<Pair, u64>, tokens: &[String]) -> Option<Pair> {
    pairs
        .iter()
        .max_by(|(pair_a, count_a), (pair_b, count_b)| {
            count_a.cmp(count_b).then_with(|| {
                let a = (&tokens[pair_a.0 as usize], &tokens[pair_a.1 as usize]);
                let b = (&tokens[pair_b.0 as usize], &tokens[pair_b.1 as usize]);
                a.cmp(&b)
            })
            // Dwa różne tokeny mogą mieć ten sam napis, a kolejność w `HashMap` jest losowa.
            .then_with(|| pair_a.cmp(pair_b))
        })
        .map(|(pair, _)| *pair)
}

/// Skleja parę w jednym słowie, w miejscu i od lewej: w „aaa” para (a, a) scala się raz.
fn merge_in_word(symbols: &mut Vec<u32>, pair: Pair, new_id: u32) {
    let mut read = 0;
    let mut write = 0;
    while read < symbols.len() {
        if read + 1 < symbols.len() && symbols[read] == pair.0 && symbols[read + 1] == pair.1 {
            symbols[write] = new_id;
            read += 2;
        } else {
            symbols[write] = symbols[read];
            read += 1;
        }
        write += 1;
    }
    symbols.truncate(write);
}

/// Czasy faz treningu w sekundach, żeby było widać, co ogranicza zysk z wątków.
#[derive(IntoPyObject)]
struct Phases {
    count: f64,
    select: f64,
    merge: f64,
    /// Suma czasu wszystkich łączeń map, po wszystkich wątkach (zawiera się w `count`).
    reduce_cpu: f64,
    reduce_calls: u64,
    pairs_last_round: usize,
}

/// Wynik treningu: (alfabet, scalenia jako pary napisów, sekundy pętli treningu, fazy).
type TrainOutput = (Vec<String>, Vec<(String, String)>, f64, Phases);

/// Uczy się listy scaleń na liście `(kawałek tekstu, liczba wystąpień)`.
///
/// Symbolami startowymi są znaki, jak w wariancie „słowa + znaki”.
#[pyfunction]
#[pyo3(signature = (chunks, num_merges, num_threads=1))]
fn train(
    py: Python<'_>,
    chunks: Vec<(String, u64)>,
    num_merges: usize,
    num_threads: usize,
) -> PyResult<TrainOutput> {
    // Zwalniamy blokadę Pythona (GIL) na czas obliczeń: tu nie dotykamy obiektów Pythona.
    py.detach(|| {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(pool.install(|| run_training(&chunks, num_merges)))
    })
}

/// Zamienia kawałki tekstu na numery tokenów. Alfabet jest posortowany, żeby numery były
/// powtarzalne między uruchomieniami.
fn prepare(chunks: &[(String, u64)]) -> (Vec<String>, Vec<Word>) {
    let alphabet: Vec<String> = chunks
        .iter()
        .flat_map(|(chunk, _)| chunk.chars())
        .collect::<BTreeSet<char>>()
        .into_iter()
        .map(String::from)
        .collect();
    let ids: HashMap<&str, u32> = alphabet
        .iter()
        .enumerate()
        .map(|(id, symbol)| (symbol.as_str(), id as u32))
        .collect();

    let words: Vec<Word> = chunks
        .iter()
        .map(|(chunk, count)| Word {
            symbols: chunk
                .chars()
                .map(|c| ids[c.encode_utf8(&mut [0; 4]) as &str])
                .collect(),
            count: *count,
        })
        .collect();
    (alphabet, words)
}

fn run_training(chunks: &[(String, u64)], num_merges: usize) -> TrainOutput {
    let (alphabet, mut words) = prepare(chunks);

    // Numer tokenu → jego tekst. Każde scalenie dopisuje nowy token na końcu.
    let mut tokens = alphabet.clone();
    let mut merges = Vec::with_capacity(num_merges);

    let reduce_nanos = AtomicU64::new(0);
    let reduce_calls = AtomicU64::new(0);
    let mut phases = Phases {
        count: 0.0,
        select: 0.0,
        merge: 0.0,
        reduce_cpu: 0.0,
        reduce_calls: 0,
        pairs_last_round: 0,
    };

    let start = Instant::now();
    for _ in 0..num_merges {
        let t = Instant::now();
        let pairs = count_pairs(&words, &reduce_nanos, &reduce_calls);
        phases.count += t.elapsed().as_secs_f64();
        phases.pairs_last_round = pairs.len();

        let t = Instant::now();
        let Some(pair) = most_frequent_pair(&pairs, &tokens) else {
            break; // każde słowo jest już jednym tokenem
        };
        phases.select += t.elapsed().as_secs_f64();

        let t = Instant::now();
        let new_id = tokens.len() as u32;
        words
            .par_iter_mut()
            .for_each(|word| merge_in_word(&mut word.symbols, pair, new_id));
        phases.merge += t.elapsed().as_secs_f64();

        let (left, right) = (&tokens[pair.0 as usize], &tokens[pair.1 as usize]);
        merges.push((left.clone(), right.clone()));
        tokens.push(format!("{left}{right}"));
    }
    phases.reduce_cpu = reduce_nanos.load(Ordering::Relaxed) as f64 / 1e9;
    phases.reduce_calls = reduce_calls.load(Ordering::Relaxed);
    (alphabet, merges, start.elapsed().as_secs_f64(), phases)
}

/// Dodaje wkład jednego słowa do liczników par i do indeksu „para → słowa”.
fn add_word(
    word_id: usize,
    word: &Word,
    counts: &mut HashMap<Pair, u64>,
    index: &mut HashMap<Pair, HashSet<usize>>,
) {
    for window in word.symbols.windows(2) {
        let pair = (window[0], window[1]);
        *counts.entry(pair).or_insert(0) += word.count;
        index.entry(pair).or_default().insert(word_id);
    }
}

/// Odejmuje wkład jednego słowa. Pary z zerowym licznikiem znikają, żeby nigdy nie wygrały
/// wyboru, tak jak w wersji naiwnej, która widzi tylko pary faktycznie obecne w danych.
fn remove_word(
    word_id: usize,
    word: &Word,
    counts: &mut HashMap<Pair, u64>,
    index: &mut HashMap<Pair, HashSet<usize>>,
) {
    for window in word.symbols.windows(2) {
        let pair = (window[0], window[1]);
        if let Some(count) = counts.get_mut(&pair) {
            *count -= word.count;
            if *count == 0 {
                counts.remove(&pair);
            }
        }
        if let Some(word_ids) = index.get_mut(&pair) {
            word_ids.remove(&word_id);
            if word_ids.is_empty() {
                index.remove(&pair);
            }
        }
    }
}

/// Ta sama lista scaleń co `train`, ale pary liczymy od zera tylko raz, na starcie.
///
/// Scalenie zmienia wyłącznie słowa zawierające scalaną parę, a indeks mówi, które to słowa.
/// Dla każdego z nich odejmujemy cały jego wkład, scalamy i dodajemy wkład od nowa: to
/// najprostsza poprawna aktualizacja, bo nie trzeba rozważać pojedynczych sąsiadów.
#[pyfunction]
fn train_indexed(py: Python<'_>, chunks: Vec<(String, u64)>, num_merges: usize) -> TrainOutput {
    py.detach(|| run_indexed_training(&chunks, num_merges))
}

fn run_indexed_training(chunks: &[(String, u64)], num_merges: usize) -> TrainOutput {
    let (alphabet, mut words) = prepare(chunks);
    let mut tokens = alphabet.clone();
    let mut merges = Vec::with_capacity(num_merges);
    let mut phases = Phases {
        count: 0.0,
        select: 0.0,
        merge: 0.0,
        reduce_cpu: 0.0,
        reduce_calls: 0,
        pairs_last_round: 0,
    };

    let start = Instant::now();
    let mut counts = HashMap::new();
    let mut index = HashMap::new();
    for (word_id, word) in words.iter().enumerate() {
        add_word(word_id, word, &mut counts, &mut index);
    }
    phases.count = start.elapsed().as_secs_f64();

    for _ in 0..num_merges {
        let t = Instant::now();
        let Some(pair) = most_frequent_pair(&counts, &tokens) else {
            break; // każde słowo jest już jednym tokenem
        };
        phases.select += t.elapsed().as_secs_f64();
        phases.pairs_last_round = counts.len();

        let t = Instant::now();
        let new_id = tokens.len() as u32;
        // Kopia listy, bo `remove_word` zmienia indeks, po którym byśmy iterowali.
        let affected: Vec<usize> = index[&pair].iter().copied().collect();
        for word_id in affected {
            remove_word(word_id, &words[word_id], &mut counts, &mut index);
            merge_in_word(&mut words[word_id].symbols, pair, new_id);
            add_word(word_id, &words[word_id], &mut counts, &mut index);
        }
        phases.merge += t.elapsed().as_secs_f64();

        let (left, right) = (&tokens[pair.0 as usize], &tokens[pair.1 as usize]);
        merges.push((left.clone(), right.clone()));
        tokens.push(format!("{left}{right}"));
    }
    (alphabet, merges, start.elapsed().as_secs_f64(), phases)
}

/// Stan korpusu z licznikami. Każda zmiana słowa przechodzi przez `rewrite`, więc liczniki par,
/// bieżące liczności tokenów i suma tokenów nie mogą się rozjechać.
///
/// Tej samej struktury używa trening i odtwarzanie na innym tekście, więc oba robią to samo.
struct Corpus {
    words: Vec<Word>,
    counts: HashMap<Pair, u64>,
    index: HashMap<Pair, HashSet<usize>>,
    /// Token → słowa, w których stoi. Potrzebne do rozłożenia tokenu bez przeglądania korpusu.
    token_words: Vec<HashSet<usize>>,
    /// Bieżąca liczność `L`: ile razy token stoi teraz w korpusie.
    live: Vec<u64>,
    total_tokens: u64,
}

impl Corpus {
    fn new(words: Vec<Word>, num_tokens: usize) -> Self {
        let mut corpus = Corpus {
            words,
            counts: HashMap::new(),
            index: HashMap::new(),
            token_words: vec![HashSet::new(); num_tokens],
            live: vec![0; num_tokens],
            total_tokens: 0,
        };
        for word_id in 0..corpus.words.len() {
            corpus.add_contribution(word_id);
        }
        corpus
    }

    fn add_contribution(&mut self, word_id: usize) {
        let word = &self.words[word_id];
        add_word(word_id, word, &mut self.counts, &mut self.index);
        for &symbol in &word.symbols {
            self.token_words[symbol as usize].insert(word_id);
            self.live[symbol as usize] += word.count;
        }
        self.total_tokens += word.symbols.len() as u64 * word.count;
    }

    fn remove_contribution(&mut self, word_id: usize) {
        let word = &self.words[word_id];
        remove_word(word_id, word, &mut self.counts, &mut self.index);
        for &symbol in &word.symbols {
            self.token_words[symbol as usize].remove(&word_id);
            self.live[symbol as usize] -= word.count;
        }
        self.total_tokens -= word.symbols.len() as u64 * word.count;
    }

    fn rewrite(&mut self, word_id: usize, change: impl FnOnce(&mut Vec<u32>)) {
        self.remove_contribution(word_id);
        change(&mut self.words[word_id].symbols);
        self.add_contribution(word_id);
    }

    fn add_token_slot(&mut self) {
        self.token_words.push(HashSet::new());
        self.live.push(0);
    }

    fn merge(&mut self, pair: Pair, new_id: u32) {
        let affected: Vec<usize> = match self.index.get(&pair) {
            Some(word_ids) => word_ids.iter().copied().collect(),
            None => return, // na innym tekście tej pary może w ogóle nie być
        };
        for word_id in affected {
            self.rewrite(word_id, |symbols| merge_in_word(symbols, pair, new_id));
        }
    }

    /// Zastępuje każde wystąpienie tokenu jego częściami.
    fn split(&mut self, token: u32, parts: &[u32]) {
        let affected: Vec<usize> = self.token_words[token as usize].iter().copied().collect();
        for word_id in affected {
            self.rewrite(word_id, |symbols| {
                let mut expanded = Vec::with_capacity(symbols.len() + parts.len());
                for &symbol in symbols.iter() {
                    if symbol == token {
                        expanded.extend_from_slice(parts);
                    } else {
                        expanded.push(symbol);
                    }
                }
                *symbols = expanded;
            });
        }
    }
}

/// Rozkłada token na części, które są teraz w słowniku.
///
/// Rodzic mógł zostać usunięty wcześniej, więc czasem trzeba zejść głębiej. Znaki bazowe są zawsze
/// żywe, dlatego rekurencja zawsze się kończy i każdy tekst da się zapisać.
fn expand(token: u32, parents: &[Option<Pair>], alive: &[bool], out: &mut Vec<u32>) {
    if alive[token as usize] {
        out.push(token);
    } else {
        let (left, right) = parents[token as usize].expect("znak bazowy jest zawsze żywy");
        expand(left, parents, alive, out);
        expand(right, parents, alive, out);
    }
}

/// Kiedy usuwać token, któremu scalenie właśnie zabrało wystąpienia.
#[derive(Clone, Copy)]
enum PruneRule {
    /// Bieżąca liczność spadła poniżej połowy liczności z chwili wejścia do słownika.
    Half,
    /// Usunięcie się opłaca: koszt `L × (części − 1)` jest mniejszy niż zysk `f_best`.
    Best,
}

fn parse_prune_rule(name: Option<&str>) -> PyResult<Option<PruneRule>> {
    match name {
        None => Ok(None),
        Some("half") => Ok(Some(PruneRule::Half)),
        Some("best") => Ok(Some(PruneRule::Best)),
        Some(other) => Err(PyValueError::new_err(format!(
            "nieznana reguła usuwania: {other} (dozwolone: half, best albo None)"
        ))),
    }
}

/// Wynik treningu ze zdarzeniami. W Pythonie to słownik o tych samych kluczach.
#[derive(IntoPyObject)]
struct EventOutput {
    /// Numer tokenu → tekst. Na początku znaki bazowe, potem tokeny w kolejności powstania.
    tokens: Vec<String>,
    /// Numer tokenu → para rodziców; `None` dla znaków bazowych.
    parents: Vec<Option<Pair>>,
    /// Historia w kolejności wykonania: `("merge", token)` albo `("split", token)`.
    events: Vec<(&'static str, u32)>,
    /// Tokeny, które są w słowniku na końcu treningu.
    final_vocab: Vec<u32>,
    vocab_size: usize,
    total_tokens: u64,
    merges: usize,
    splits: usize,
    /// Ile scaleń odtworzyło token, który był już wcześniej usunięty.
    restores: usize,
    /// Czy po wszystkich zdarzeniach każde słowo nadal skleja się do swojego tekstu.
    text_preserved: bool,
    /// Czy trening przerwał bezpiecznik zamiast osiągnąć docelowy rozmiar słownika.
    hit_merge_cap: bool,
    seconds: f64,
}

/// Trening do docelowego rozmiaru słownika, z opcjonalnym usuwaniem tokenów.
///
/// `prune=None` to zwykłe BPE, `"half"` i `"best"` to dwie reguły usuwania.
#[pyfunction]
#[pyo3(signature = (chunks, target_vocab, prune=None))]
fn train_events(
    py: Python<'_>,
    chunks: Vec<(String, u64)>,
    target_vocab: usize,
    prune: Option<String>,
) -> PyResult<EventOutput> {
    let rule = parse_prune_rule(prune.as_deref())?;
    Ok(py.detach(|| run_event_training(&chunks, target_vocab, rule)))
}

fn should_prune(
    token: u32,
    merged_count: u64,
    rule: PruneRule,
    corpus: &Corpus,
    parents: &[Option<Pair>],
    alive: &[bool],
    entry_count: &[u64],
) -> bool {
    let live = corpus.live[token as usize];
    match rule {
        PruneRule::Half => live * 2 < entry_count[token as usize],
        PruneRule::Best => {
            // Szukanie maksimum przegląda wszystkie pary, więc najpierw tani filtr: `f_best` po
            // scaleniu zwykle nie przekracza liczności tego scalenia.
            if live >= merged_count {
                return false;
            }
            let (left, right) = parents[token as usize].expect("znaków bazowych nie usuwamy");
            let mut parts = Vec::new();
            expand(left, parents, alive, &mut parts);
            expand(right, parents, alive, &mut parts);
            let cost = live * (parts.len() as u64 - 1);
            let best = corpus.counts.values().copied().max().unwrap_or(0);
            cost < best
        }
    }
}

fn run_event_training(
    chunks: &[(String, u64)],
    target_vocab: usize,
    rule: Option<PruneRule>,
) -> EventOutput {
    let start = Instant::now();
    let (alphabet, words) = prepare(chunks);
    let originals: Vec<&str> = chunks.iter().map(|(chunk, _)| chunk.as_str()).collect();
    let alphabet_size = alphabet.len();

    let mut corpus = Corpus::new(words, alphabet_size);
    let mut tokens = alphabet;
    let mut parents: Vec<Option<Pair>> = vec![None; alphabet_size];
    let mut alive = vec![true; alphabet_size];
    let mut entry_count = vec![0u64; alphabet_size];
    // Ta sama para zawsze daje ten sam token, więc odtworzony token wraca pod swoim numerem.
    let mut pair_ids: HashMap<Pair, u32> = HashMap::new();

    let mut events = Vec::new();
    let mut vocab = alphabet_size;
    let (mut merges, mut splits, mut restores) = (0, 0, 0);
    // Bezpiecznik: reguła bez gwarancji spadku mogłaby w teorii usuwać i odtwarzać bez końca.
    let merge_cap = target_vocab.saturating_mul(20);

    while vocab < target_vocab && merges < merge_cap {
        let Some(pair) = most_frequent_pair(&corpus.counts, &tokens) else {
            break; // każde słowo jest już jednym tokenem
        };
        let id = match pair_ids.get(&pair) {
            Some(&id) => id,
            None => {
                let id = tokens.len() as u32;
                pair_ids.insert(pair, id);
                tokens.push(format!("{}{}", tokens[pair.0 as usize], tokens[pair.1 as usize]));
                parents.push(Some(pair));
                alive.push(false);
                entry_count.push(0);
                corpus.add_token_slot();
                id
            }
        };

        let before = corpus.live[id as usize];
        corpus.merge(pair, id);
        let merged_count = corpus.live[id as usize] - before;
        events.push(("merge", id));
        merges += 1;
        if !alive[id as usize] {
            // `before > 0` nie może tu wystąpić: martwy token nie stoi w korpusie.
            restores += usize::from(entry_count[id as usize] > 0);
            alive[id as usize] = true;
            entry_count[id as usize] = corpus.live[id as usize];
            vocab += 1;
        }

        // Liczność spada tylko rodzicom tego scalenia, więc tylko ich trzeba sprawdzać.
        let Some(rule) = rule else { continue };
        let candidates = if pair.0 == pair.1 { vec![pair.0] } else { vec![pair.0, pair.1] };
        for token in candidates {
            // Znaki bazowe zostają zawsze: dzięki nim da się zapisać każdy tekst.
            if parents[token as usize].is_none() || !alive[token as usize] {
                continue;
            }
            if should_prune(token, merged_count, rule, &corpus, &parents, &alive, &entry_count) {
                alive[token as usize] = false;
                vocab -= 1;
                let (left, right) = parents[token as usize].expect("sprawdzone wyżej");
                let mut parts = Vec::new();
                expand(left, &parents, &alive, &mut parts);
                expand(right, &parents, &alive, &mut parts);
                corpus.split(token, &parts);
                events.push(("split", token));
                splits += 1;
            }
        }
    }

    let text_preserved = corpus.words.iter().zip(&originals).all(|(word, original)| {
        let text: String = word.symbols.iter().map(|&s| tokens[s as usize].as_str()).collect();
        text == *original
    });
    let final_vocab = (0..tokens.len() as u32).filter(|&t| alive[t as usize]).collect();
    EventOutput {
        tokens,
        parents,
        events,
        final_vocab,
        vocab_size: vocab,
        total_tokens: corpus.total_tokens,
        merges,
        splits,
        restores,
        text_preserved,
        hit_merge_cap: merges >= merge_cap && vocab < target_vocab,
        seconds: start.elapsed().as_secs_f64(),
    }
}

/// Odtwarza historię treningu na dowolnym tekście i zapisuje liczbę tokenów w wybranych punktach.
///
/// Pomiar pada, gdy słownik pierwszy raz osiąga dany rozmiar, po usunięciach z tej samej rundy.
/// Zwraca wiersze `(rozmiar słownika, tokeny w tekście, scalenia, usunięcia)`; ostatni opisuje stan
/// końcowy.
#[pyfunction]
fn replay_curve(
    py: Python<'_>,
    chunks: Vec<(String, u64)>,
    tokens: Vec<String>,
    parents: Vec<Option<Pair>>,
    events: Vec<(String, u32)>,
    vocab_points: Vec<usize>,
) -> PyResult<Vec<(usize, u64, usize, usize)>> {
    py.detach(|| {
        let mut ids: HashMap<String, u32> = HashMap::new();
        let mut alive = Vec::with_capacity(tokens.len());
        let mut base_count = 0;
        for (id, parent) in parents.iter().enumerate() {
            alive.push(parent.is_none());
            if parent.is_none() {
                ids.insert(tokens[id].clone(), id as u32);
                base_count += 1;
            }
        }

        // Znak spoza alfabetu treningu nigdy się nie scali, więc zostaje osobnym tokenem. Dostaje
        // numer poza słownikiem i nie zwiększa jego rozmiaru.
        let mut next_unknown = tokens.len() as u32;
        let words: Vec<Word> = chunks
            .iter()
            .map(|(chunk, count)| Word {
                symbols: chunk
                    .chars()
                    .map(|c| {
                        *ids.entry(c.to_string()).or_insert_with(|| {
                            let id = next_unknown;
                            next_unknown += 1;
                            id
                        })
                    })
                    .collect(),
                count: *count,
            })
            .collect();
        let mut full_parents = parents.clone();
        full_parents.resize(next_unknown as usize, None);
        alive.resize(next_unknown as usize, true);

        let mut corpus = Corpus::new(words, next_unknown as usize);
        let points: HashSet<usize> = vocab_points.into_iter().collect();
        let mut vocab = base_count;
        let mut largest = base_count;
        let (mut merges, mut splits) = (0, 0);
        let mut rows = Vec::new();
        if points.contains(&vocab) {
            rows.push((vocab, corpus.total_tokens, merges, splits));
        }

        for (position, (kind, token)) in events.iter().enumerate() {
            match kind.as_str() {
                "merge" => {
                    let pair = full_parents[*token as usize].ok_or_else(|| {
                        PyValueError::new_err(format!("token {token} nie ma rodziców"))
                    })?;
                    corpus.merge(pair, *token);
                    if !alive[*token as usize] {
                        alive[*token as usize] = true;
                        vocab += 1;
                    }
                    merges += 1;
                }
                "split" => {
                    alive[*token as usize] = false;
                    vocab -= 1;
                    let (left, right) = full_parents[*token as usize].ok_or_else(|| {
                        PyValueError::new_err(format!("token {token} nie ma rodziców"))
                    })?;
                    let mut parts = Vec::new();
                    expand(left, &full_parents, &alive, &mut parts);
                    expand(right, &full_parents, &alive, &mut parts);
                    corpus.split(*token, &parts);
                    splits += 1;
                }
                other => {
                    return Err(PyValueError::new_err(format!("nieznane zdarzenie: {other}")));
                }
            }
            // Runda to scalenie razem z usunięciami, które po nim nastąpiły.
            let round_ended = events.get(position + 1).is_none_or(|(next, _)| next == "merge");
            if round_ended && vocab > largest {
                largest = vocab;
                if points.contains(&vocab) {
                    rows.push((vocab, corpus.total_tokens, merges, splits));
                }
            }
        }
        // Ostatni wiersz to zawsze stan po całej historii, nawet gdy trening skończył się wcześniej,
        // bo zabrakło par, a słownik był wtedy mniejszy niż w szczycie.
        let last = (vocab, corpus.total_tokens, merges, splits);
        if rows.last() != Some(&last) {
            rows.push(last);
        }
        Ok(rows)
    })
}

#[pymodule]
fn bpe_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(train, m)?)?;
    m.add_function(wrap_pyfunction!(train_indexed, m)?)?;
    m.add_function(wrap_pyfunction!(train_events, m)?)?;
    m.add_function(wrap_pyfunction!(replay_curve, m)?)?;
    Ok(())
}
