//! Benchmark: insert ~250 diverse memories, run queries, report similarity distributions.
//! Usage: cargo run --bin bench_search
//!
//! Run from the project root (where models/ directory is).

use std::path::Path;
use std::path::PathBuf;
use std::time::Instant;

use anima_core::memory::Memory;
use anima_core::namespace::Namespace;
use anima_core::search::{ScorerConfig, SearchMode};
use anima_db::pool::DbPool;
use anima_db::store::MemoryStore;
use anima_embed::model::{EmbeddingModel, PoolingStrategy};

/// ~250 diverse memories spanning many topics
fn memory_corpus() -> Vec<(&'static str, &'static str, Vec<&'static str>)> {
    vec![
        // ─── Food & Diet (20) ───
        ("User loves banana", "preference", vec!["food", "fruit"]),
        ("User loves bananas", "preference", vec!["food"]),
        ("User is vegetarian", "preference", vec!["food", "diet"]),
        ("User prefers spicy food", "preference", vec!["food"]),
        ("User's favorite cuisine is Japanese", "preference", vec!["food", "cuisine"]),
        ("User is allergic to peanuts", "fact", vec!["food", "health"]),
        ("User drinks coffee every morning", "preference", vec!["food", "routine"]),
        ("User prefers tea over coffee", "preference", vec!["food", "drinks"]),
        ("User likes cooking Italian food", "preference", vec!["food", "cooking"]),
        ("User avoids dairy products", "preference", vec!["food", "diet"]),
        ("User's favorite fruit is mango", "preference", vec!["food", "fruit"]),
        ("User enjoys baking sourdough bread", "preference", vec!["food", "cooking"]),
        ("User is on a keto diet", "fact", vec!["food", "diet"]),
        ("User likes sushi", "preference", vec!["food"]),
        ("User hates cilantro", "preference", vec!["food"]),
        ("User prefers oat milk", "preference", vec!["food", "drinks"]),
        ("User eats lunch at noon every day", "fact", vec!["food", "routine"]),
        ("User's wife Sarah loves cooking Italian food", "fact", vec!["family", "food"]),
        ("User's wife Sarah is allergic to shellfish", "fact", vec!["family", "health"]),
        ("User makes homemade pasta on Sundays", "fact", vec!["food", "routine"]),

        // ─── Family & Relationships (15) ───
        ("User is married to Sarah", "fact", vec!["family"]),
        ("User has two children", "fact", vec!["family"]),
        ("User's daughter is named Emma", "fact", vec!["family"]),
        ("User's son is named Jack", "fact", vec!["family"]),
        ("User's mother lives in Florida", "fact", vec!["family", "location"]),
        ("User's father is retired", "fact", vec!["family"]),
        ("User has a brother named Mike", "fact", vec!["family"]),
        ("User's wife works as a nurse", "fact", vec!["family", "work"]),
        ("User's parents are divorced", "fact", vec!["family"]),
        ("User's sister lives in London", "fact", vec!["family", "location"]),
        ("User and wife met in college", "fact", vec!["family", "history"]),
        ("User's wife Sarah prefers sushi", "fact", vec!["family", "food"]),
        ("User's in-laws visit every Christmas", "fact", vec!["family", "events"]),
        ("User coaches son's little league team", "fact", vec!["family", "sports"]),
        ("User drives the kids to school every morning", "fact", vec!["family", "routine"]),

        // ─── Work & Career (20) ───
        ("User works at Google", "fact", vec!["work"]),
        ("User is a software engineer", "fact", vec!["work", "career"]),
        ("User's team uses React and TypeScript", "fact", vec!["work", "tech"]),
        ("User works remotely from home", "fact", vec!["work"]),
        ("User's manager is named Lisa", "fact", vec!["work"]),
        ("User has been at Google for 3 years", "fact", vec!["work"]),
        ("User previously worked at Amazon", "fact", vec!["work", "history"]),
        ("User is considering changing jobs", "decision", vec!["work", "career"]),
        ("User earns $180k per year", "fact", vec!["work", "finance"]),
        ("User works on the search team", "fact", vec!["work"]),
        ("User has a standup meeting at 9am daily", "fact", vec!["work", "routine"]),
        ("User prefers VS Code over IntelliJ", "preference", vec!["work", "tools"]),
        ("User uses vim keybindings", "preference", vec!["work", "tools"]),
        ("User got promoted to senior engineer", "event", vec!["work", "career"]),
        ("User's team is building a new microservice", "fact", vec!["work", "tech"]),
        ("User prefers trunk-based development", "preference", vec!["work", "tech"]),
        ("User uses Docker for local development", "fact", vec!["work", "tech"]),
        ("User's team has 8 people", "fact", vec!["work"]),
        ("User mentors two junior engineers", "fact", vec!["work"]),
        ("User dislikes meetings without agendas", "preference", vec!["work"]),

        // ─── Technology & Programming (20) ───
        ("User is learning Rust", "fact", vec!["tech", "learning"]),
        ("User prefers Python for scripting", "preference", vec!["tech"]),
        ("User uses Linux as primary OS", "preference", vec!["tech"]),
        ("User has a home server running Proxmox", "fact", vec!["tech"]),
        ("User built a personal website with Next.js", "fact", vec!["tech", "projects"]),
        ("User prefers dark mode in all apps", "preference", vec!["tech"]),
        ("User uses Neovim as text editor", "preference", vec!["tech", "tools"]),
        ("User prefers PostgreSQL over MySQL", "preference", vec!["tech", "databases"]),
        ("User is interested in machine learning", "fact", vec!["tech", "interests"]),
        ("User built a chatbot using GPT-4", "fact", vec!["tech", "projects"]),
        ("User uses Arch Linux on desktop", "fact", vec!["tech"]),
        ("User prefers functional programming", "preference", vec!["tech"]),
        ("User uses tmux for terminal multiplexing", "preference", vec!["tech", "tools"]),
        ("User prefers GraphQL over REST", "preference", vec!["tech"]),
        ("User is learning about distributed systems", "fact", vec!["tech", "learning"]),
        ("User uses Tailwind CSS for styling", "preference", vec!["tech"]),
        ("User prefers SQLite for small projects", "preference", vec!["tech", "databases"]),
        ("User has experience with Kubernetes", "fact", vec!["tech"]),
        ("User's favorite programming language is Rust", "preference", vec!["tech"]),
        ("User prefers mechanical keyboards", "preference", vec!["tech"]),

        // ─── Location & Travel (15) ───
        ("User lives in San Francisco", "fact", vec!["location"]),
        ("User recently moved to Seattle", "event", vec!["location"]),
        ("User grew up in Chicago", "fact", vec!["location", "history"]),
        ("User wants to visit Japan", "preference", vec!["travel"]),
        ("User traveled to Italy last summer", "event", vec!["travel"]),
        ("User's favorite city is Tokyo", "preference", vec!["travel", "location"]),
        ("User hates long flights", "preference", vec!["travel"]),
        ("User has a trip planned to Paris in March", "event", vec!["travel"]),
        ("User prefers Airbnb over hotels", "preference", vec!["travel"]),
        ("User has visited 20 countries", "fact", vec!["travel"]),
        ("User commutes by bike", "fact", vec!["location", "routine"]),
        ("User lives in a two-bedroom apartment", "fact", vec!["location", "home"]),
        ("User is thinking about buying a house", "decision", vec!["location", "finance"]),
        ("User's office is in downtown Seattle", "fact", vec!["work", "location"]),
        ("User lived in New York for 5 years", "fact", vec!["location", "history"]),

        // ─── Hobbies & Interests (20) ───
        ("User plays guitar", "fact", vec!["hobbies", "music"]),
        ("User enjoys hiking on weekends", "preference", vec!["hobbies", "outdoors"]),
        ("User reads science fiction books", "preference", vec!["hobbies", "reading"]),
        ("User watches anime", "preference", vec!["hobbies", "entertainment"]),
        ("User plays chess competitively", "fact", vec!["hobbies"]),
        ("User's favorite author is Isaac Asimov", "preference", vec!["hobbies", "reading"]),
        ("User collects vinyl records", "fact", vec!["hobbies", "music"]),
        ("User runs 5km every morning", "fact", vec!["hobbies", "fitness"]),
        ("User practices yoga twice a week", "fact", vec!["hobbies", "fitness"]),
        ("User enjoys woodworking", "fact", vec!["hobbies"]),
        ("User is training for a marathon", "event", vec!["hobbies", "fitness"]),
        ("User plays video games on PS5", "fact", vec!["hobbies", "gaming"]),
        ("User's favorite game is Elden Ring", "preference", vec!["hobbies", "gaming"]),
        ("User watches Premier League football", "preference", vec!["hobbies", "sports"]),
        ("User supports Arsenal FC", "preference", vec!["hobbies", "sports"]),
        ("User enjoys photography", "fact", vec!["hobbies"]),
        ("User paints watercolors", "fact", vec!["hobbies", "art"]),
        ("User is learning to play piano", "fact", vec!["hobbies", "music"]),
        ("User listens to jazz music", "preference", vec!["hobbies", "music"]),
        ("User plays Dungeons and Dragons on Fridays", "fact", vec!["hobbies"]),

        // ─── Health & Fitness (15) ───
        ("User is 26 years old", "fact", vec!["personal"]),
        ("User weighs 75 kg", "fact", vec!["health"]),
        ("User is 180 cm tall", "fact", vec!["health"]),
        ("User wears glasses", "fact", vec!["health"]),
        ("User has mild asthma", "fact", vec!["health"]),
        ("User takes vitamin D supplements", "fact", vec!["health"]),
        ("User sleeps 7 hours per night", "fact", vec!["health", "routine"]),
        ("User has a gym membership", "fact", vec!["health", "fitness"]),
        ("User broke his arm last year", "event", vec!["health"]),
        ("User is trying to lose weight", "decision", vec!["health"]),
        ("User does intermittent fasting", "fact", vec!["health", "diet"]),
        ("User has a standing desk", "fact", vec!["health", "work"]),
        ("User meditates daily", "fact", vec!["health", "routine"]),
        ("User's blood type is O positive", "fact", vec!["health"]),
        ("User quit smoking two years ago", "event", vec!["health"]),

        // ─── Finance (15) ───
        ("User invests in index funds", "fact", vec!["finance"]),
        ("User uses a budget app", "fact", vec!["finance", "tools"]),
        ("User is saving for a house down payment", "decision", vec!["finance"]),
        ("User has student loans", "fact", vec!["finance"]),
        ("User prefers shopping online", "preference", vec!["shopping"]),
        ("User drives a Toyota Camry", "fact", vec!["personal"]),
        ("User is considering buying a Tesla", "decision", vec!["personal", "finance"]),
        ("User pays rent of $2500 per month", "fact", vec!["finance", "home"]),
        ("User has a credit score of 780", "fact", vec!["finance"]),
        ("User's 401k is with Fidelity", "fact", vec!["finance"]),
        ("User donates to charity monthly", "fact", vec!["finance"]),
        ("User subscribes to Netflix and Spotify", "fact", vec!["entertainment"]),
        ("User bought a new laptop recently", "event", vec!["shopping", "tech"]),
        ("User started investing in cryptocurrency", "event", vec!["finance"]),
        ("User's lease expires in September", "fact", vec!["home", "finance"]),

        // ─── Pets (10) ───
        ("User has a dog named Max", "fact", vec!["pets"]),
        ("User's dog Max is a golden retriever", "fact", vec!["pets"]),
        ("User adopted Max from a shelter", "event", vec!["pets"]),
        ("User takes Max to the park every evening", "fact", vec!["pets", "routine"]),
        ("User has a cat named Luna", "fact", vec!["pets"]),
        ("User's cat Luna is 3 years old", "fact", vec!["pets"]),
        ("User feeds pets organic food", "preference", vec!["pets", "food"]),
        ("User spends $200/month on pet care", "fact", vec!["pets", "finance"]),
        ("User wants to adopt another dog", "decision", vec!["pets"]),
        ("User's dog Max needs hip surgery", "event", vec!["pets", "health"]),

        // ─── Education (10) ───
        ("User has a BS in Computer Science from MIT", "fact", vec!["education"]),
        ("User graduated in 2020", "fact", vec!["education"]),
        ("User is considering getting an MBA", "decision", vec!["education"]),
        ("User took online courses on Coursera", "fact", vec!["education", "learning"]),
        ("User studied abroad in Japan during college", "fact", vec!["education", "travel"]),
        ("User was on the dean's list", "fact", vec!["education"]),
        ("User's favorite subject was algorithms", "preference", vec!["education"]),
        ("User tutored math in college", "fact", vec!["education"]),
        ("User reads technical papers weekly", "fact", vec!["education", "routine"]),
        ("User attends tech conferences annually", "fact", vec!["education", "work"]),

        // ─── Social & Personal (20) ───
        ("User prefers texting over calling", "preference", vec!["social"]),
        ("User is introverted", "fact", vec!["personal"]),
        ("User has a small close friend group", "fact", vec!["social"]),
        ("User deleted Facebook account", "event", vec!["social"]),
        ("User hosts dinner parties monthly", "fact", vec!["social", "food"]),
        ("User's best friend is named Tom", "fact", vec!["social"]),
        ("User volunteers at a local food bank", "fact", vec!["social"]),
        ("User is a morning person", "preference", vec!["personal", "routine"]),
        ("User wakes up at 6am", "fact", vec!["personal", "routine"]),
        ("User prefers minimalist design", "preference", vec!["personal"]),
        ("User values privacy highly", "preference", vec!["personal"]),
        ("User is left-handed", "fact", vec!["personal"]),
        ("User's favorite color is blue", "preference", vec!["personal"]),
        ("User prefers summer over winter", "preference", vec!["personal"]),
        ("User's birthday is on March 15", "fact", vec!["personal"]),
        ("User's wedding anniversary is June 20", "fact", vec!["family"]),
        ("User speaks fluent Japanese", "fact", vec!["personal", "skills"]),
        ("User has dual citizenship US and UK", "fact", vec!["personal"]),
        ("User learned to code at age 12", "fact", vec!["personal", "tech"]),
        ("User's favorite movie is Inception", "preference", vec!["entertainment"]),

        // ─── Miscellaneous (10) ───
        ("User has a podcast about tech", "fact", vec!["tech", "hobbies"]),
        ("User brews beer as a hobby", "fact", vec!["hobbies"]),
        ("User collects Lego sets", "fact", vec!["hobbies"]),
        ("User has a home office setup with two monitors", "fact", vec!["work", "tech"]),
        ("User reads the Hacker News daily", "fact", vec!["tech", "routine"]),
        ("User has a Raspberry Pi cluster at home", "fact", vec!["tech", "hobbies"]),
        ("User writes a weekly blog post", "fact", vec!["hobbies", "writing"]),
        ("User prefers window seats on planes", "preference", vec!["travel"]),
        ("User has a vegetable garden", "fact", vec!["hobbies", "food"]),
        ("User's wife Sarah is a nurse at Seattle General", "fact", vec!["family", "work"]),
    ]
}

/// Test queries — some should match, some should not
fn test_queries() -> Vec<(&'static str, Vec<&'static str>)> {
    vec![
        // Direct keyword matches
        ("wife", vec!["wife", "sarah", "married"]),
        ("banana", vec!["banana"]),
        ("dog", vec!["dog", "max"]),
        ("programming language", vec!["rust", "python", "programming", "language"]),
        ("favorite food", vec!["food", "cuisine", "sushi", "banana", "mango", "cilantro"]),
        ("cat", vec!["cat", "luna"]),
        ("children kids", vec!["daughter", "son", "children", "kids", "emma", "jack"]),

        // Semantic (no keyword overlap expected)
        ("spouse", vec!["wife", "married", "sarah"]),
        ("partner", vec!["wife", "married", "sarah"]),
        ("exercise routine", vec!["run", "yoga", "gym", "marathon", "fitness", "hiking"]),
        ("salary compensation", vec!["earns", "180k"]),
        ("pet animal", vec!["dog", "cat", "max", "luna", "pet"]),
        ("where does user live", vec!["lives", "francisco", "seattle", "apartment"]),
        ("coding tools editor", vec!["vs code", "neovim", "vim", "tmux", "docker"]),
        ("education background", vec!["mit", "computer science", "graduated", "college"]),
        ("investment money", vec!["index fund", "401k", "saving", "invest", "fidelity"]),
        ("health condition", vec!["asthma", "glasses", "weight", "arm", "blood"]),
        ("hobbies free time", vec!["guitar", "hiking", "chess", "yoga", "woodworking", "photography"]),

        // Queries that should return NO relevant results
        ("quantum physics experiments", vec![]),
        ("ancient Egyptian pharaoh burial", vec![]),
        ("deep sea submarine exploration", vec![]),
        ("nuclear fusion reactor design", vec![]),
        ("medieval castle siege warfare", vec![]),
        ("black hole event horizon", vec![]),
        ("photosynthesis chlorophyll", vec![]),
        ("tectonic plate movement earthquake", vec![]),
    ]
}

fn resolve_model_dir() -> anyhow::Result<PathBuf> {
    if let Ok(dir) = std::env::var("ANIMA_BENCH_MODEL_DIR") {
        let p = PathBuf::from(dir);
        if p.join("model.onnx").exists() && p.join("tokenizer.json").exists() {
            return Ok(p);
        }
        anyhow::bail!(
            "ANIMA_BENCH_MODEL_DIR is set but missing model.onnx/tokenizer.json: {}",
            p.display()
        );
    }

    // Prefer the current project default.
    let candidates = [Path::new("models/qwen3-embedding-0.6b"), Path::new("models/bge-m3")];
    for c in candidates {
        if c.join("model.onnx").exists() && c.join("tokenizer.json").exists() {
            return Ok(c.to_path_buf());
        }
    }

    anyhow::bail!(
        "Could not find benchmark model dir. Tried models/qwen3-embedding-0.6b and models/bge-m3"
    )
}

fn resolve_dimension(model_dir: &Path) -> usize {
    if let Ok(raw) = std::env::var("ANIMA_BENCH_EMBED_DIM") {
        if let Ok(parsed) = raw.parse::<usize>() {
            return parsed.max(1);
        }
    }
    if model_dir
        .file_name()
        .and_then(|v| v.to_str())
        .map(|v| v.contains("qwen3-embedding"))
        .unwrap_or(false)
    {
        512
    } else {
        DbPool::DEFAULT_DIMENSION
    }
}

fn resolve_pooling(model_dir: &Path) -> PoolingStrategy {
    if let Ok(raw) = std::env::var("ANIMA_BENCH_EMBED_POOLING") {
        if let Ok(strategy) = PoolingStrategy::parse(&raw) {
            return strategy;
        }
    }
    if model_dir
        .file_name()
        .and_then(|v| v.to_str())
        .map(|v| v.contains("qwen3-embedding"))
        .unwrap_or(false)
    {
        PoolingStrategy::LastToken
    } else {
        PoolingStrategy::Mean
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("=== Search Similarity Benchmark ===\n");

    // Load embedding model
    let model_dir = resolve_model_dir()?;
    let model_path = model_dir.join("model.onnx");
    let tokenizer_path = model_dir.join("tokenizer.json");
    let dimension = resolve_dimension(&model_dir);
    let pooling = resolve_pooling(&model_dir);

    println!("Loading embedding model from {}...", model_dir.display());
    println!("Embedding config: dim={}, pooling={}", dimension, pooling.as_str());
    let embedder = EmbeddingModel::load(&model_path, &tokenizer_path, dimension, pooling, None, None)?;

    // Create in-memory store
    let (pool, _) = DbPool::open(":memory:", dimension)?;
    let store = MemoryStore::new(pool);

    // Generate corpus
    let corpus = memory_corpus();
    println!("Inserting {} memories...\n", corpus.len());

    let start = Instant::now();

    // Batch embed for speed
    let texts: Vec<&str> = corpus.iter().map(|(c, _, _)| *c).collect();
    let embeddings = embedder.embed_batch(&texts)?;

    for (i, (content, mem_type, tags)) in corpus.iter().enumerate() {
        let memory = Memory::new(
            "bench/default".into(),
            content.to_string(),
            None,
            tags.iter().map(|t| t.to_string()).collect(),
            Some(mem_type.to_string()),
        );
        store.insert(&memory, &embeddings[i]).await?;
    }
    println!("Inserted {} memories in {:.1}s\n", corpus.len(), start.elapsed().as_secs_f64());

    // Run queries using the store's search (which uses the same path as the real app)
    let ns = Namespace::parse("bench/default")?;
    let queries = test_queries();

    // We need raw vector distances to analyze thresholds.
    // Use vector-only mode with a large limit to get everything.
    let large_config = ScorerConfig {
        temporal_weight: 0.0, // disable temporal for clean analysis
        ..ScorerConfig::default()
    };

    println!("{:<40} {:>5}  Top 5 results (+ = relevant, - = irrelevant)", "Query", "Hits");
    println!("{}", "-".repeat(140));

    let mut relevant_scores: Vec<f64> = Vec::new();
    let mut irrelevant_scores: Vec<f64> = Vec::new();

    for (query, expected_keywords) in &queries {
        let query_emb = embedder.embed_query(query)?;

        // Vector-only search with large limit to see score distribution
        let results = store
            .search(&query_emb, &anima_embed::SparseVector::default(), query, &ns, &SearchMode::Vector, 200, &large_config)
            .await?;

        let mut top_display: Vec<String> = Vec::new();

        for r in &results {
            let mem = store.get(&r.memory_id).await?;
            if let Some(m) = mem {
                let content_lower = m.content.to_lowercase();
                let is_relevant = if expected_keywords.is_empty() {
                    false
                } else {
                    expected_keywords.iter().any(|kw| content_lower.contains(&kw.to_lowercase()))
                };

                // vector_score is the normalized RRF score, but we want raw similarity
                // The score field is what we use for thresholding
                let score = r.score;

                if is_relevant {
                    relevant_scores.push(score);
                } else {
                    irrelevant_scores.push(score);
                }

                if top_display.len() < 5 {
                    let marker = if is_relevant { "+" } else { "-" };
                    let truncated = if m.content.len() > 45 { format!("{}...", &m.content[..42]) } else { m.content.clone() };
                    top_display.push(format!("[{marker}{score:.3}] {truncated}"));
                }
            }
        }

        println!("{:<40} {:>5}  {}", query, results.len(), top_display.join("  |  "));
    }

    // Now do the same but with raw L2 distances to analyze the actual similarity values
    println!("\n\n=== Raw Cosine Similarity Analysis (from L2 distance) ===\n");
    println!("Computing raw similarities for each query...\n");

    let mut raw_relevant: Vec<f64> = Vec::new();
    let mut raw_irrelevant: Vec<f64> = Vec::new();

    // Build a map of content -> embedding for raw distance computation
    let corpus_embeddings: Vec<(&str, Vec<f32>)> = texts.iter().zip(embeddings.iter())
        .map(|(t, e)| (*t, e.clone()))
        .collect();

    for (query, expected_keywords) in &queries {
        let query_emb = embedder.embed_query(query)?;

        for (content, emb) in &corpus_embeddings {
            // Compute cosine similarity (embeddings are L2-normalized, so dot product = cosine sim)
            let sim: f64 = query_emb.iter().zip(emb.iter())
                .map(|(a, b)| (*a as f64) * (*b as f64))
                .sum();

            let content_lower = content.to_lowercase();
            let is_relevant = if expected_keywords.is_empty() {
                false
            } else {
                expected_keywords.iter().any(|kw| content_lower.contains(&kw.to_lowercase()))
            };

            if is_relevant {
                raw_relevant.push(sim);
            } else {
                raw_irrelevant.push(sim);
            }
        }
    }

    // Sort for percentile analysis
    raw_relevant.sort_by(|a, b| b.partial_cmp(a).unwrap());
    raw_irrelevant.sort_by(|a, b| b.partial_cmp(a).unwrap());

    println!("Relevant results: {}", raw_relevant.len());
    println!("  Max: {:.4}  Min: {:.4}  Mean: {:.4}  P25: {:.4}  P50: {:.4}  P75: {:.4}",
        raw_relevant.first().copied().unwrap_or(0.0),
        raw_relevant.last().copied().unwrap_or(0.0),
        raw_relevant.iter().sum::<f64>() / raw_relevant.len().max(1) as f64,
        raw_relevant.get(raw_relevant.len() / 4).copied().unwrap_or(0.0),
        raw_relevant.get(raw_relevant.len() / 2).copied().unwrap_or(0.0),
        raw_relevant.get(3 * raw_relevant.len() / 4).copied().unwrap_or(0.0),
    );
    println!("\nIrrelevant results: {}", raw_irrelevant.len());
    println!("  Max: {:.4}  Min: {:.4}  Mean: {:.4}  P25: {:.4}  P50: {:.4}  P75: {:.4}",
        raw_irrelevant.first().copied().unwrap_or(0.0),
        raw_irrelevant.last().copied().unwrap_or(0.0),
        raw_irrelevant.iter().sum::<f64>() / raw_irrelevant.len().max(1) as f64,
        raw_irrelevant.get(raw_irrelevant.len() / 4).copied().unwrap_or(0.0),
        raw_irrelevant.get(raw_irrelevant.len() / 2).copied().unwrap_or(0.0),
        raw_irrelevant.get(3 * raw_irrelevant.len() / 4).copied().unwrap_or(0.0),
    );

    // Histogram
    println!("\n\n=== Cosine Similarity Distribution ===\n");
    let buckets: Vec<f64> = (-2..=10).map(|i| i as f64 * 0.1).collect();
    println!("{:<15} {:>10} {:>10}", "Range", "Relevant", "Irrelevant");
    println!("{}", "-".repeat(40));

    for i in 0..buckets.len() - 1 {
        let lo = buckets[i];
        let hi = buckets[i + 1];
        let rel = raw_relevant.iter().filter(|&&s| s >= lo && s < hi).count();
        let irr = raw_irrelevant.iter().filter(|&&s| s >= lo && s < hi).count();
        if rel > 0 || irr > 0 {
            let rel_bar = "#".repeat((rel as f64 / 10.0).ceil() as usize);
            let irr_bar = ".".repeat((irr as f64 / 50.0).ceil() as usize);
            println!("[{:>5.2}, {:>5.2})  {:>6} {:<20} {:>6} {}", lo, hi, rel, rel_bar, irr, irr_bar);
        }
    }

    // Threshold analysis
    println!("\n\n=== Threshold Analysis (Cosine Similarity) ===\n");
    println!("{:<12} {:>6} {:>6} {:>6} {:>8} {:>8} {:>8}", "Threshold", "TP", "FP", "FN", "Prec", "Recall", "F1");
    println!("{}", "-".repeat(65));

    let total_rel = raw_relevant.len();
    for &thresh in &[0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60] {
        let tp = raw_relevant.iter().filter(|&&s| s >= thresh).count();
        let fp = raw_irrelevant.iter().filter(|&&s| s >= thresh).count();
        let fn_ = total_rel - tp;

        let precision = if tp + fp > 0 { tp as f64 / (tp + fp) as f64 } else { 0.0 };
        let recall = if total_rel > 0 { tp as f64 / total_rel as f64 } else { 0.0 };
        let f1 = if precision + recall > 0.0 { 2.0 * precision * recall / (precision + recall) } else { 0.0 };

        println!("{:<12.2} {:>6} {:>6} {:>6} {:>8.3} {:>8.3} {:>8.3}", thresh, tp, fp, fn_, precision, recall, f1);
    }

    // Per-query breakdown for "unrelated" queries
    println!("\n\n=== Unrelated Query Max Similarities ===");
    println!("(These should ideally all be below the chosen threshold)\n");

    for (query, expected_keywords) in &queries {
        if !expected_keywords.is_empty() { continue; }

        let query_emb = embedder.embed_query(query)?;
        let mut max_sim: f64 = 0.0;
        let mut max_content = String::new();

        for (content, emb) in &corpus_embeddings {
            let sim: f64 = query_emb.iter().zip(emb.iter())
                .map(|(a, b)| (*a as f64) * (*b as f64))
                .sum();
            if sim > max_sim {
                max_sim = sim;
                max_content = content.to_string();
            }
        }

        let truncated = if max_content.len() > 50 { format!("{}...", &max_content[..47]) } else { max_content };
        println!("  {:<40} max_sim={:.4}  closest: {}", query, max_sim, truncated);
    }

    // ═══════════════════════════════════════════════════════════════════
    // Graph Neighbor Expansion Benchmark
    // ═══════════════════════════════════════════════════════════════════

    println!("\n\n=== Graph Neighbor Expansion Benchmark ===\n");
    println!("Simulates RAG pipeline: search top-5 → expand via graph neighbors → measure quality\n");

    // Queries that benefit from graph expansion (related facts spread across memories)
    let graph_queries: Vec<(&str, Vec<&str>, Vec<&str>)> = vec![
        // (query, keywords for direct hits, keywords for expanded hits we WANT to find)
        ("wife",
         vec!["wife", "married", "sarah"],
         vec!["sarah", "nurse", "italian", "sushi", "shellfish", "college"]),
        ("dog",
         vec!["dog", "max"],
         vec!["max", "retriever", "shelter", "park", "pet", "hip surgery"]),
        ("cat",
         vec!["cat", "luna"],
         vec!["luna", "pet", "organic"]),
        ("where does user work",
         vec!["google", "software engineer", "remote", "search team"],
         vec!["react", "typescript", "manager", "lisa", "standup", "promoted", "senior"]),
        ("exercise routine",
         vec!["run", "yoga", "gym", "marathon", "hiking"],
         vec!["5km", "morning", "fitness", "training", "weight"]),
        ("children kids",
         vec!["daughter", "son", "children", "emma", "jack"],
         vec!["school", "little league", "family"]),
        ("programming language",
         vec!["rust", "python", "programming", "language"],
         vec!["learning", "scripting", "functional"]),
        ("investment money savings",
         vec!["index fund", "401k", "saving", "invest", "fidelity", "cryptocurrency"],
         vec!["budget", "house", "down payment", "student loan"]),
    ];

    // Test different parameter combos
    let neighbor_limits = [2, 3, 5];
    let min_sims = [0.3, 0.4, 0.5, 0.6, 0.7];
    let search_top_k = 5;
    let max_total: usize = 12;

    println!("Parameters: search_top_k={search_top_k}, max_total={max_total}\n");

    // Header for parameter sweep
    println!("{:<12} {:>6}  {:>6}  {:>6}  {:>6}  {:>8}  {:>8}  {:>8}  {:>8}",
        "nbr_sim", "n_lim", "direct", "graph", "total", "d_rel", "g_rel", "g_noise", "g_prec");
    println!("{}", "-".repeat(95));

    for &min_sim in &min_sims {
        for &nbr_limit in &neighbor_limits {
            let mut total_direct = 0usize;
            let mut total_graph = 0usize;
            let mut total_direct_relevant = 0usize;
            let mut total_graph_relevant = 0usize;
            let mut total_graph_noise = 0usize;

            for (query, direct_kws, expanded_kws) in &graph_queries {
                let query_emb = embedder.embed_query(query)?;

                // Phase 1: Search top-K (using hybrid mode like the real pipeline)
                let search_results = store
                    .search(&query_emb, &anima_embed::SparseVector::default(), query, &ns, &SearchMode::Hybrid, search_top_k, &large_config)
                    .await?;

                let mut seen_ids: std::collections::HashSet<String> = std::collections::HashSet::new();
                let mut direct_relevant = 0usize;

                for sr in &search_results {
                    seen_ids.insert(sr.memory_id.clone());
                    if let Some(m) = store.get(&sr.memory_id).await? {
                        let lower = m.content.to_lowercase();
                        if direct_kws.iter().any(|kw| lower.contains(&kw.to_lowercase())) ||
                           expanded_kws.iter().any(|kw| lower.contains(&kw.to_lowercase())) {
                            direct_relevant += 1;
                        }
                    }
                }

                // Phase 2: Graph expansion
                let mut graph_neighbors: Vec<(String, String, f64)> = Vec::new(); // (id, content, sim)
                let remaining = max_total.saturating_sub(search_results.len());

                for sr in &search_results {
                    if graph_neighbors.len() >= remaining { break; }
                    let neighbors = store.find_neighbors(&sr.memory_id, nbr_limit, min_sim).await?;
                    for (mem, sim) in neighbors {
                        if seen_ids.contains(&mem.id) { continue; }
                        if graph_neighbors.len() >= remaining { break; }
                        seen_ids.insert(mem.id.clone());
                        graph_neighbors.push((mem.id, mem.content, sim));
                    }
                }

                let mut graph_relevant = 0usize;
                let mut graph_noise = 0usize;
                for (_id, content, _sim) in &graph_neighbors {
                    let lower = content.to_lowercase();
                    if direct_kws.iter().any(|kw| lower.contains(&kw.to_lowercase())) ||
                       expanded_kws.iter().any(|kw| lower.contains(&kw.to_lowercase())) {
                        graph_relevant += 1;
                    } else {
                        graph_noise += 1;
                    }
                }

                total_direct += search_results.len();
                total_graph += graph_neighbors.len();
                total_direct_relevant += direct_relevant;
                total_graph_relevant += graph_relevant;
                total_graph_noise += graph_noise;
            }

            let graph_precision = if total_graph > 0 {
                total_graph_relevant as f64 / total_graph as f64
            } else { 0.0 };

            println!("{:<12.2} {:>6}  {:>6}  {:>6}  {:>6}  {:>8}  {:>8}  {:>8}  {:>8.3}",
                min_sim, nbr_limit,
                total_direct, total_graph, total_direct + total_graph,
                total_direct_relevant, total_graph_relevant, total_graph_noise, graph_precision);
        }
    }

    // Detailed per-query breakdown with the current production params
    let prod_min_sim = 0.5;
    let prod_nbr_limit = 3;

    println!("\n\n=== Per-Query Graph Expansion Detail (min_sim={prod_min_sim}, nbr_limit={prod_nbr_limit}) ===\n");

    for (query, direct_kws, expanded_kws) in &graph_queries {
        let query_emb = embedder.embed_query(query)?;

        let search_results = store
            .search(&query_emb, &anima_embed::SparseVector::default(), query, &ns, &SearchMode::Hybrid, search_top_k, &large_config)
            .await?;

        println!("Query: \"{query}\"");
        println!("  Direct hits ({}):", search_results.len());

        let mut seen_ids: std::collections::HashSet<String> = std::collections::HashSet::new();
        for sr in &search_results {
            seen_ids.insert(sr.memory_id.clone());
            if let Some(m) = store.get(&sr.memory_id).await? {
                let lower = m.content.to_lowercase();
                let relevant = direct_kws.iter().any(|kw| lower.contains(&kw.to_lowercase())) ||
                               expanded_kws.iter().any(|kw| lower.contains(&kw.to_lowercase()));
                let marker = if relevant { "+" } else { "-" };
                let truncated = if m.content.len() > 60 { format!("{}...", &m.content[..57]) } else { m.content.clone() };
                println!("    [{marker} {:.3}] {}", sr.score, truncated);
            }
        }

        // Graph expansion
        let remaining = max_total.saturating_sub(search_results.len());
        let mut graph_neighbors: Vec<(String, f64)> = Vec::new();
        for sr in &search_results {
            if graph_neighbors.len() >= remaining { break; }
            let neighbors = store.find_neighbors(&sr.memory_id, prod_nbr_limit, prod_min_sim).await?;
            for (mem, sim) in neighbors {
                if seen_ids.contains(&mem.id) { continue; }
                if graph_neighbors.len() >= remaining { break; }
                seen_ids.insert(mem.id.clone());
                let lower = mem.content.to_lowercase();
                let relevant = direct_kws.iter().any(|kw| lower.contains(&kw.to_lowercase())) ||
                               expanded_kws.iter().any(|kw| lower.contains(&kw.to_lowercase()));
                let marker = if relevant { "+" } else { "-" };
                let truncated = if mem.content.len() > 60 { format!("{}...", &mem.content[..57]) } else { mem.content.clone() };
                println!("    [{marker} g:{sim:.3}] {truncated}");
                graph_neighbors.push((mem.id, sim));
            }
        }

        if graph_neighbors.is_empty() {
            println!("  Graph neighbors: (none above threshold)");
        } else {
            println!("  Graph neighbors: {}", graph_neighbors.len());
        }
        println!();
    }

    // Test graph expansion on unrelated queries (should add no noise)
    println!("\n=== Graph Expansion on Unrelated Queries (should add minimal noise) ===\n");

    for (query, expected_keywords) in &queries {
        if !expected_keywords.is_empty() { continue; }

        let query_emb = embedder.embed_query(query)?;
        let search_results = store
            .search(&query_emb, &anima_embed::SparseVector::default(), query, &ns, &SearchMode::Hybrid, search_top_k, &large_config)
            .await?;

        let mut seen_ids: std::collections::HashSet<String> = std::collections::HashSet::new();
        for sr in &search_results {
            seen_ids.insert(sr.memory_id.clone());
        }

        let remaining = max_total.saturating_sub(search_results.len());
        let mut graph_count = 0usize;
        for sr in &search_results {
            if graph_count >= remaining { break; }
            let neighbors = store.find_neighbors(&sr.memory_id, prod_nbr_limit, prod_min_sim).await?;
            for (mem, _sim) in neighbors {
                if seen_ids.contains(&mem.id) { continue; }
                if graph_count >= remaining { break; }
                seen_ids.insert(mem.id.clone());
                graph_count += 1;
            }
        }

        println!("  {:<40} direct={:<3} graph_expanded=+{}", query, search_results.len(), graph_count);
    }

    Ok(())
}
