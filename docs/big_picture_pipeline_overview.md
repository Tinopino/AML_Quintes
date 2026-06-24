# Big Picture Pipeline Overview

This document explains what the car-insurance policy summary pipeline does, why it is built this way, and how someone can check the result.

It is written for a company contact who needs to understand the work without reading the Python code.

## Short Version

The pipeline turns long Dutch car-insurance policy PDFs into short, clickable visual summaries.

The final summary is not just a free-written AI summary. Each displayed statement keeps a link back to the original policy text, the page number, and the marked PDF. This makes the result easier to review and easier to trust.

The current final output is a browser page for each policy, plus the original PDF and an annotated PDF.

## Why Not Ask AI To Summarize The Whole PDF?

A car-insurance policy is long, detailed, and full of tables. A direct AI summary can sound clear while still missing an important rule or mixing up whether something is covered or not covered.

The pipeline avoids that by splitting the work into smaller checks:

1. Read the PDF and keep the structure.
2. Break the policy into small source statements.
3. Ask the AI model to find possible useful facts.
4. Check that each quote really appears in the policy.
5. Keep the most useful customer-facing facts.
6. Build a visual summary with links back to the source.

The goal is not only a nice summary. The goal is a summary that can be checked.

## What Goes In

The input is a folder of Dutch car-insurance policy PDFs.

In the current run, the source folder contains 40 policy documents.

## What Comes Out

For each policy, the final package contains:

- A browser summary page, named `policy_summary.html`.
- The original PDF.
- An annotated PDF with source locations marked.
- A data file with the final summary items, named `filtered_summary_items.json`.

Across the current final run:

- 40 policies were processed.
- 1,073 PDF pages were read.
- 22,400 small source statements were created.
- 16,625 possible facts were found by the AI model.
- 1,933 final items are displayed in the visual summaries.
- Every displayed quote in the final summary has a source quote, page, and source text link recorded.

These numbers describe the current run. If the input PDFs or model settings change, the numbers can change.

## One-Line Flow

```text
PDFs -> cleaned policy text -> small source statements -> possible facts -> source checks -> concise visual summary
```

## Step-By-Step Walkthrough

## Step 1: Read The PDF

The pipeline starts with the original PDF files. It reads the words, page layout, headings, lists, and tables.

This matters because insurance meaning is often stored in layout. For example, a table column can mean "covered" while another column can mean "not covered". If we only keep the words and lose the table position, the meaning can change.

What this step keeps:

- Document name.
- Page number.
- Text on the page.
- Table and list structure.
- Location of text on the PDF page.

## Step 2: Remove Repeated Page Noise

Policies often repeat the same headers, footers, page numbers, and branding on many pages.

The pipeline removes repeated page noise so that the later steps focus on policy rules instead of repeated decoration.

The important rule is that useful policy text should stay. Only repeated noise should be removed.

## Step 3: Keep The Heading Trail

The same sentence can mean different things depending on where it appears.

For example, a rule under "All Risk" should not be treated the same as a rule under "WA". WA means third-party liability cover. Beperkt Casco means limited own-damage cover. All Risk means broader own-damage cover.

The pipeline keeps the heading trail around each piece of text, so later steps know which part of the policy it belongs to.

## Step 4: Split The Policy Into Small Source Statements

A policy paragraph or table cell can contain more than one rule. The pipeline splits the text into smaller statements that can be checked one by one.

Each small source statement keeps:

- The policy document it came from.
- The page it came from.
- The heading trail around it.
- The table role when the text came from a table, such as covered or not covered.
- A link back to the source text.

This is the main reason the system can later explain where a summary item came from.

## Step 5: Ask The AI Model To Find Possible Useful Facts

The AI model reads small groups of nearby source statements. It looks for facts that may be useful in a customer summary.

It looks for:

- What is covered.
- What is not covered.
- What the customer must report.
- What the customer must do after damage.
- Deadlines.
- Money limits.
- Deductibles.
- Conditions that affect cover.

At this stage, the system keeps too much on purpose. It is better to keep extra candidates here than to miss an important policy rule too early.

## Step 6: Check The Source Quotes

For every possible fact, the AI model must give a source quote.

The pipeline checks whether that quote really appears in the policy text. It also links the item back to the source page and the small source statement.

This check does not prove that every legal interpretation is perfect. It proves that the displayed item is tied to real policy text, so a human reviewer can inspect it.

## Step 7: Fix Common Table Mistakes

Insurance tables can be tricky. A row can be under a column that means "covered" or under a column that means "not covered".

The pipeline uses the saved table role to reduce mistakes where a covered item is treated as an exclusion, or an exclusion is treated as covered.

## Step 8: Build A Broad First Summary

The system groups the checked facts by customer topic.

Examples:

- Cover by module, such as WA, Beperkt Casco, and All Risk.
- Common exclusions.
- Exclusions that only apply to one module.
- Notification duties.
- Claim duties.
- Limits and deductibles.
- Deadlines.
- Conditions.

This first grouped list is still too long for a customer-facing summary. Its job is to make sure useful facts are available for the final selection step.

## Step 9: Keep The Most Useful Customer-Facing Items

A second AI model step selects and merges items from the checked list.

This step is not allowed to invent new policy facts. It can only choose from already checked items.

The selection gives priority to facts that a customer most needs to see before or while using the insurance, such as:

- Theft.
- Fire.
- Storm and hail.
- Window damage.
- Parking damage.
- Collision with an animal.
- Alcohol and drugs.
- Driving licence duties.
- Reporting duties.
- Claim deadlines.
- Important limits and deductibles.

The current target is usually 35 to 80 grouped items per policy. This keeps the summary readable while still showing the important policy rules.

## Step 10: Build The Visual Summary

The final page-building step creates a browser page for each policy.

The page groups items into tabs and cards. Each card shows a plain headline, a short explanation, and the source quote. The user can open the linked page in the annotated PDF to check the policy text.

The final package also includes an index page, so the company can open all 40 summaries from one place.

## What The Final Summary Shows

The summary is arranged around the questions a customer is likely to have:

- What is covered?
- What is not covered?
- What must I report?
- What must I do after damage?
- What limits or deductibles apply?
- Which deadlines matter?
- Which conditions can affect cover?

The summary is meant to support review and customer explanation. It is not a replacement for the legal policy document.

## How The Result Can Be Checked

Each final item can be checked in three ways:

- Read the displayed source quote.
- Open the source page in the PDF.
- Inspect the saved data file for the document, page, and source text link.

This is why the pipeline stores source links throughout the process.

## Current Quality Checks

The current run includes several checks.

Source quote check:

- 16,551 possible facts had a direct quote match.
- 53 possible facts had a close quote match.
- 21 possible facts needed weaker matching or stayed unverified.
- This means 16,604 out of 16,625 possible facts had a direct or close source quote match.

Final summary trace check:

- The final summaries contain 4,931 source quote records.
- All 4,931 have a source quote, source page, and source text link recorded.

The most important interpretation is this: the pipeline makes the result checkable. A reviewer can see where a statement came from and decide whether the final customer wording is acceptable.

## Where AI Is Used

AI is used in two places:

- First, to find possible useful facts in the policy text.
- Second, to select and merge the checked facts into a shorter customer-facing summary.

The local code handles PDF reading, text splitting, source quote checks, table role handling, and rendering.

The AI provider cost depends on the number and length of policies and on the selected model prices.

## What This Pipeline Is Good At

- Turning long PDFs into a reviewable customer summary.
- Keeping links back to the source policy text.
- Handling headings, tables, and different cover modules.
- Making AI output easier to inspect.
- Producing consistent outputs across many policy documents.

## Known Limits

- The tool supports human review. It does not replace legal or compliance approval.
- Some definitions or administrative clauses can still enter the candidate list before filtering.
- Similar clauses can appear more than once when policies repeat the same rule in multiple places.
- PDF text quality depends on the source PDF. Poorly extracted text can reduce quality.
- The current final visual output is a browser page and annotated PDF.
- Any privacy or data-sharing requirements must be checked before sending policy text to an external AI provider.

## Decisions For The Company

The company should decide:

- Whether the final deliverable should remain a browser page or also receive another export format.
- How short or detailed each customer summary should be.
- Which policy topics are mandatory in every summary.
- Who signs off on legal wording.
- Whether policy text may be sent to the selected AI provider.
- How the summaries should fit into the company review workflow.

## Main Files For Developers

Most readers can stop before this section. These files are listed so the work can be reproduced.

Main input:

```text
data/raw/car_policies/*.pdf
```

Main output folders from the current run:

```text
outputs/current_full_gpt4omini
outputs/filter_gpt41_context_tags_fixed
outputs/final_policy_summaries
```

Main run commands:

```bash
python -m src.pipeline.run_policy_summary_pipeline --all --output-dir outputs/current_full_gpt4omini --model gpt-4o-mini --cache-dir outputs/current_full_gpt4omini/cache_context_tags --max-concurrent 20
python -m src.pipeline.run_summary_filter --input-dir outputs/current_full_gpt4omini --output-dir outputs/filter_gpt41_context_tags_fixed --all --model gpt-4.1 --max-concurrent 8 --fail-on-error
python -m src.rendering.policy_summary --input-dir outputs/filter_gpt41_context_tags_fixed --output-dir outputs/final_policy_summaries --all --pdf-root data/raw/car_policies --no-open
```

Useful documents:

```text
README.md
docs/big_picture_pipeline_overview.md
docs/policy_summary_pipeline_reproducibility.md
```
