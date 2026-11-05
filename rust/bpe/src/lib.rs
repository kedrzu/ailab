//! Trening BPE w Ruście: ten sam naiwny algorytm co `src/ailab/bpe.py`.
//!
//! Każda runda liczy wszystkie pary od nowa, bez indeksu „para → słowa”. Zmieniają się tylko
//! język, zapis tokenów jako liczb oraz wątki, żeby dało się zmierzyć zysk z każdej z tych rzeczy.

use std::collections::{BTreeSet, HashMap};
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

fn run_training(chunks: &[(String, u64)], num_merges: usize) -> TrainOutput {
    // Alfabet posortowany, żeby numery tokenów były powtarzalne między uruchomieniami.
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

    let mut words: Vec<Word> = chunks
        .iter()
        .map(|(chunk, count)| Word {
            symbols: chunk
                .chars()
                .map(|c| ids[c.encode_utf8(&mut [0; 4]) as &str])
                .collect(),
            count: *count,
        })
        .collect();

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

#[pymodule]
fn bpe_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(train, m)?)?;
    Ok(())
}
