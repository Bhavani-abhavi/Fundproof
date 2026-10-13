# Walkthrough: the innovations (interview questions by innovation)

Written so that both owners (Rahul Vajja and Bhavani Adula) can explain each innovation in their own words. The
plan is in `docs/innovations.md`. Each innovation also has a longer walkthrough with every file and design choice:

| Innovation | Long walkthrough | Status of its numbers |
|---|---|---|
| S1: local-model generation eval and LLM migration gate | `docs/walkthrough_innovations_s1.md` | see that file |
| S2: containment over detection (injection attack success by layer) | `docs/walkthrough_innovations_s2.md` | attack success not measured yet (run R4 has not run) |

No number in this file comes from memory. Every number is copied from a results file named next to it. Any
attack-success number will only exist in `results/injection_metrics.json` after the full run (R4).

## S2: containment over detection

**In one paragraph.** An underwriting copilot reads text the applicant wrote. The applicant wants approval, so that
text is where a prompt injection would come from. S2 measures how often an injected instruction still reaches the
answer under three prompt layers. L0 is naive pasting, L1 is the current prompt, and L2 is a separate, normalized
applicant block with its own id namespace. The set is 60 frozen payloads: 40 public, 20 hand-written. S2 also proves
with tests that a fully obedient model still cannot change a decision.

**Numbers available now (no model involved).** These come from `results/injection/injection_metrics_detector_only.json`,
written by `make injection-detector`. The tripwire (`agent.injection.detect`) flagged:
- 12 of 60 payloads on their raw text (Wilson 95% 0.1183-0.3178);
- 16 of 60 after normalization (0.1713-0.3901);
- 7 of the 40 public payloads either way;
- 2 of 96 clean policy chunks (NBC-9.2 and HM-11.2, which quote injection phrases on purpose).

### Interview questions and answers

1. **What is the difference between direct and indirect prompt injection?**
   - Direct injection is typed by the person using the chat.
   - Indirect injection arrives inside data the system reads: a document, an email, a retrieved snippet.
   - Here the underwriter is the user and the applicant writes the documents, so the attack is indirect. The
     attacker never talks to the model; they only write text the model will read.

2. **Why is the applicant the attacker?**
   - The applicant controls the notes, the uploaded documents and even the names on the application, and benefits
     from approval.
   - So everything they write is a claim to verify, never policy and never an instruction.
   - In the L2 prompt their text sits in its own `<untrusted_applicant_document id="APP-1">` block, and a rule says
     exactly that.

3. **Why is the detector only a tripwire?**
   - It is a fixed list of patterns.
   - On the 40 public payloads, written by other people, it flagged 7, so most phrasings pass.
   - A flag is useful for routing: it adds a blocking referral to Fraud Operations, and it can never move a file
     toward approval.
   - The protection has to come from somewhere else.
   - The S2 review also found that the product agent ran the tripwire on the raw text only. A zero-width space
     inside "ignore" hid the instruction. The agent now checks the raw text and its normalized form, and a test
     proves the product flags every payload the normalized detector flags.

4. **What does "containment by architecture" mean, and what does "0 by construction" cover?** Even a model that
   obeys every injected instruction can only change words in an answer or a memo:
   - no tool on the allow-list can approve, decline or clear;
   - the LLM may request only the read-only guideline search;
   - the recommendation comes from deterministic code;
   - the memo checker rejects decision wording and a mismatched recommendation;
   - only a human at the checkpoint decides.

   `tests/security/test_containment.py` checks all of this over every payload. "0 by construction" is about the
   model's influence on the decision. It is not a claim that applicant text can never matter. The review found one
   place where it did: a sanctioned name written with an invisible character used to slip past sanctions screening.
   Screening now normalizes the screened name, and a test runs each obfuscation through the real matcher code.

5. **What is ASCII smuggling, and what else hides text?**
   - Unicode tag characters (U+E0000-U+E007F) mirror ASCII but render as nothing, so an instruction can hide
     where no reviewer sees it.
   - The same family includes zero-width characters, the soft hyphen, bidirectional controls, variation
     selectors (used to hide bytes after an emoji) and Hangul fillers.
   - `security/normalize.py` strips the whole Unicode Default_Ignorable_Code_Point set, taken from the Unicode
     14.0.0 data file. It also folds lookalike letters and angle-bracket lookalikes.
   - The tripwire also reads the decoded tag message, so a hidden instruction is checked rather than ignored.

6. **Why report seen and unseen payloads separately, and how unseen are the public ones?**
   - The author wrote the hand-written payloads knowing the defenses, which flatters the results, so they are
     reported apart.
   - The public payloads are unseen for the tripwire: other people wrote them and no pattern was tuned on them.
   - For attack success they are only partly third-party. Each one ends with one of four goal sentences the author
     wrote, and the canary is reached only by following that sentence. Some public rows carry no instruction of
     their own.
   - 11 of the 40 public payloads fall into 5 groups of near-duplicates (computed in the results file), so the
     public intervals are somewhat too narrow.

7. **Why is "100% blocked" not a claim?**
   - The payload set is small and fixed, and the model is one 3B local model.
   - An adaptive attacker would try new phrasings.
   - The scorer counts fixed canary patterns, so the attack success rate is a lower bound for paraphrases.
   - The defensible statements are a per-layer attack success rate with a confidence interval on these payloads,
     and "decision-level impact 0 by construction, enforced by a test".

8. **How does this map to the OWASP Top 10 for LLM Applications 2025?** (Primary source: genai.owasp.org/llm-top-10/,
   checked 2026-10-13.)
   - LLM01:2025 Prompt Injection is the attack itself.
   - LLM06:2025 Excessive Agency is what the tool allow-list and the human checkpoint prevent.
   - LLM05:2025 Improper Output Handling covers the exfiltration canary (a markdown image a renderer would fetch).
   - LLM09:2025 Misinformation covers the fake citation.
   - LLM08:2025 Vector and Embedding Weaknesses (a poisoned index) is a Tier 2 extension: S2 never writes to the
     index.
   - Whether any of this applies to a real institution depends on that institution and its risk profile; this is a
     demonstration, not a compliance program.

9. **What is citation laundering?**
   - It means citing the applicant's own document as the source of a rule, for example "the DSCR minimum is
     waived [APP-1]".
   - The answer looks grounded, but the source is the attacker.
   - `check_citations(..., policy_ids_only=True)` reports it separately from attack success.
   - The `APP-` namespace also covers the agent's ids for the application's structured fields, on purpose: those
     are applicant statements too, so none of them can be the source of a policy claim.

10. **If attack success drops from L1 to L2, does that prove the separate applicant block works?** Not on its own.
    There are three things to check:
    - **Refusals.** A "Not found" answer holds no canary, so each layer also reports its not-found rate and
      key-fact recall under attack. Every question has its gold section in the context.
    - **L2's rule 2.** It tells the model not to copy "instructions, codes or links" from the applicant block. The
      author wrote it knowing that two of the four canaries are a code and a link, so part of a drop may come from
      that sentence rather than from the structure.
    - **The paired L1-vs-L2 test.** The results carry a paired L1-vs-L2 McNemar test next to L0-vs-L2, so S2's own
      increment is visible.

    If L0 attack success is already low, that mostly reflects the 3B model's weak instruction-following. How a
    stronger model would behave was not measured: it could follow the injections more, the rules more, or both.
