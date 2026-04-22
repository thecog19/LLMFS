//! Pre-tokenization: the first stage of GPT-2-family BPE.
//!
//! Byte-level BPE can't just run merges over raw text — it needs
//! "pre-tokens" (atomic string chunks) first, to ensure that BPE
//! merges never cross certain natural boundaries (letter vs digit
//! vs whitespace vs punctuation). Each pre-token is independently
//! byte-level-encoded and then merged through the BPE vocab.
//!
//! `llama.cpp` selects pre-tokenization behavior via the
//! `tokenizer.ggml.pre` string in the GGUF metadata. Each variant
//! corresponds to a *list* of regexes applied in sequence (not a
//! single pattern). The sequential application is the key subtlety:
//! regex #1 finds and isolates some class of atoms; regex #2 runs
//! on the remaining regions; and so on. For `smollm` (SmolLM /
//! SmolLM2), the list is:
//!
//! ```text
//! 1. \p{N}
//! 2. 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)
//! ```
//!
//! Regex #1 isolates every unicode digit as its own pre-token (no
//! multi-digit numbers, so the BPE vocab can't collapse "123" into
//! one token). Regex #2 is the classic GPT-2 alternation: English
//! contractions, letter runs (with optional leading space),
//! number runs (ditto), punctuation runs, and trailing whitespace
//! via `\s+(?!\S)`.
//!
//! # Empirically verified via `llama-tokenize` on SmolLM2-135M
//!
//! ```text
//! "Hello, world!"        → [Hello, ",", " world", !]
//! "abc123def"            → [abc, "1", "2", "3", def]
//! "The quick brown fox"  → [The, " quick", " brown", " fox"]
//! "hello   world"        → [hello, "  ", " world]
//! "don't worry"          → [don, 't, " worry]
//! "It's 3:14pm"          → [It, 's, " ", "3", :, "1", "4", pm]
//! ```

use fancy_regex::Regex;
use thiserror::Error;

/// Pre-tokenizer variant identified by the `tokenizer.ggml.pre`
/// string. Milestone A targets `smollm` (SmolLM / SmolLM2 family).
/// Other variants error cleanly.
#[derive(Debug)]
pub struct PreTokenizer {
    /// Compiled regexes, applied in sequence (see module docs).
    regexes: Vec<Regex>,
}

#[derive(Debug, Error)]
pub enum PreTokenizerError {
    #[error(
        "unsupported pre-tokenizer variant `{0}` — Milestone A implements `smollm` only"
    )]
    UnsupportedVariant(String),

    // `fancy_regex::Error` is a large enum (hundreds of bytes); box
    // it so `Result<_, PreTokenizerError>` stays small on the happy
    // path.
    #[error("regex compile failed for variant `{variant}`: {err}")]
    RegexCompile {
        variant: &'static str,
        err: Box<fancy_regex::Error>,
    },

    #[error("regex match iteration failed: {0}")]
    MatchFailure(Box<fancy_regex::Error>),
}

impl PreTokenizer {
    /// Build a pre-tokenizer for the named variant.
    ///
    /// Unknown variants return `UnsupportedVariant`; this is
    /// correct — the pre-tokenizer variants are small, explicit,
    /// and a silently-degrading fallback would produce wrong
    /// tokenization (and therefore wrong AWQ salience, wrong
    /// perplexity, ...). Fail loudly.
    pub fn new(variant: &str) -> Result<Self, PreTokenizerError> {
        let (name, patterns) = match variant {
            "smollm" => (
                "smollm",
                &[
                    r"\p{N}",
                    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)",
                ][..],
            ),
            other => return Err(PreTokenizerError::UnsupportedVariant(other.to_owned())),
        };
        let regexes = patterns
            .iter()
            .map(|p| {
                Regex::new(p).map_err(|err| PreTokenizerError::RegexCompile {
                    variant: name,
                    err: Box::new(err),
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { regexes })
    }

    /// Split `text` into pre-tokens by applying each configured
    /// regex in sequence. Each pass takes the current set of
    /// regions, finds all non-overlapping matches of that pass's
    /// regex inside each region, and produces a new set of regions
    /// where both matches *and* gaps between matches are kept (in
    /// left-to-right order). Zero-length regions are dropped.
    ///
    /// The returned slices borrow `text` directly.
    pub fn split<'a>(&self, text: &'a str) -> Result<Vec<&'a str>, PreTokenizerError> {
        if text.is_empty() {
            return Ok(Vec::new());
        }
        let mut regions: Vec<&'a str> = vec![text];
        for re in &self.regexes {
            let mut next: Vec<&'a str> = Vec::with_capacity(regions.len() * 2);
            for region in &regions {
                split_one_region(re, region, &mut next)?;
            }
            regions = next;
        }
        Ok(regions)
    }
}

/// Split `region` by all non-overlapping matches of `re`, pushing
/// both matches and between-match gaps into `out` in document order.
/// Empty slices are dropped.
fn split_one_region<'a>(
    re: &Regex,
    region: &'a str,
    out: &mut Vec<&'a str>,
) -> Result<(), PreTokenizerError> {
    let mut cursor = 0usize;
    for m in re.find_iter(region) {
        let m = m.map_err(|e| PreTokenizerError::MatchFailure(Box::new(e)))?;
        let start = m.start();
        let end = m.end();
        if start > cursor {
            out.push(&region[cursor..start]);
        }
        if end > start {
            out.push(&region[start..end]);
        }
        cursor = end;
    }
    if cursor < region.len() {
        out.push(&region[cursor..]);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn smollm_split(text: &str) -> Vec<String> {
        PreTokenizer::new("smollm")
            .unwrap()
            .split(text)
            .unwrap()
            .into_iter()
            .map(|s| s.to_owned())
            .collect()
    }

    #[test]
    fn unsupported_variant_errors() {
        assert!(matches!(
            PreTokenizer::new("bogus"),
            Err(PreTokenizerError::UnsupportedVariant(_))
        ));
    }

    #[test]
    fn empty_input_yields_empty_vec() {
        assert_eq!(smollm_split(""), Vec::<String>::new());
    }

    // ─── Behavioral equivalence with llama-tokenize on SmolLM2 ─────
    // These expected outputs were captured via
    //   llama-tokenize -m models/smollm2-135m-f16.gguf -p <input>
    // and stripped of the BPE merge step (each pre-token below is
    // what the regex pass should produce; BPE A2c then merges
    // adjacent chars within each pre-token into vocab entries).

    #[test]
    fn digit_run_isolates_each_digit() {
        assert_eq!(smollm_split("abc123def"), vec!["abc", "1", "2", "3", "def"]);
    }

    #[test]
    fn classic_hello_world_punctuation() {
        assert_eq!(
            smollm_split("Hello, world!"),
            vec!["Hello", ",", " world", "!"]
        );
    }

    #[test]
    fn letter_run_attracts_leading_space() {
        assert_eq!(
            smollm_split("The quick brown fox"),
            vec!["The", " quick", " brown", " fox"]
        );
    }

    #[test]
    fn multi_space_between_words_collapses_all_but_one_into_own_token() {
        // Three spaces between "hello" and "world": two form a
        // trailing-whitespace pre-token (\s+(?!\S) at the slice
        // boundary before "world"), the third attaches to
        // " world" as the optional-leading-space of `?\p{L}+`.
        assert_eq!(
            smollm_split("hello   world"),
            vec!["hello", "  ", " world"]
        );
    }

    #[test]
    fn leading_whitespace_is_collapsed_to_one_pretoken() {
        assert_eq!(
            smollm_split("   leading spaces"),
            vec!["  ", " leading", " spaces"]
        );
    }

    #[test]
    fn trailing_whitespace_survives_at_end_of_input() {
        // After the first-pass `\p{N}` leaves "trailing   " intact
        // (no digits), the gpt2 regex on the full slice: `\p{L}+`
        // matches "trailing" greedily, then "   " remains. The
        // final `\s+(?!\S)` matches the run because lookahead at
        // end-of-string succeeds.
        assert_eq!(smollm_split("trailing   "), vec!["trailing", "   "]);
    }

    #[test]
    fn english_contraction_apostrophe_is_own_token() {
        assert_eq!(smollm_split("don't worry"), vec!["don", "'t", " worry"]);
    }

    #[test]
    fn mixed_digits_letters_contractions_and_punctuation() {
        // The canonical small-input regression test.
        assert_eq!(
            smollm_split("It's 3:14pm"),
            vec!["It", "'s", " ", "3", ":", "1", "4", "pm"]
        );
    }

    #[test]
    fn returned_slices_borrow_from_input() {
        // Every returned slice is a subslice of the input, so
        // slice_ptr + slice_len fits inside input_ptr + input_len.
        let text = "abc123";
        let tokens = PreTokenizer::new("smollm").unwrap().split(text).unwrap();
        let text_range =
            text.as_ptr() as usize..text.as_ptr() as usize + text.len();
        for tok in tokens {
            let tok_start = tok.as_ptr() as usize;
            let tok_end = tok_start + tok.len();
            assert!(
                text_range.contains(&tok_start) && tok_end <= text_range.end,
                "slice escaped input",
            );
        }
    }
}
