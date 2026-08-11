# Change report, 2026-07-02

Files: LSE_capacity.tex, LSE_capacity_na.tex, LSE_capacity_appendices.tex.
Pre-edit snapshots: Backups/*_backup_v3-07-02-20h36.tex.
Each change applied to every file that shares the text.

## Errata (verified against scripts and CSVs)

1. N(alpha) formula: ceiling replaced by round in both statements; matches
   `N_for_alpha` in the AAAI scripts.
2. Full run range: N = 77 corrected to N = 76 at alpha = 0.20 (CSV: N_used = 76).
3. GPU memory: "50 GB" corrected to "48 GB" (RTX A6000).
4. Typo: Delta f(alpha T) corrected to Delta f(alpha, T).
5. Appendix B: "cited in the Abstract" corrected to "cited in the main text".
6. Title: "Thermodynamic approach" recapitalized to "Thermodynamic Approach"
   (AAAI mixed-case rule); also in the appendices title.

## Template cleanup (both main files)

7. Placeholder AAAI Press Staff author block replaced with
   \author{Anonymous submission} \affiliations{~}.
8. bibentry block (marked REMOVE by the template) and \iffalse author
   examples deleted.
9. Four dead commented-out paragraphs deleted (old PPS26 sentence, two old
   heatmap paragraphs, old Bulk-run description, "%with Full and Cone").

## Redundancy cuts

10. Finite-N smoothing sentence kept once (Results body); removed from the
    Fig. 3 caption; Discussion version condensed to one sentence.
11. Kramers escape mechanism kept in Theory; Results now cross-references it.
12. Equal-depth minima switching stated once (moved to the smoothness
    paragraph; removed from the definition sentence).
13. Intro: three restatements of "the retrieval basin survives" merged into
    one sentence.
14. "This proves geometrically..." sentence dropped (repeated the preceding
    claim).
15. Cone paragraph: repeated "the bulk enters through (28)" clause dropped.
16. Appendix B: "Two facts follow from the duality. First... tangency is
    automatic" dropped (re-proved the direct proof); paragraph now starts at
    the envelope-theorem derivation.

## Clarity

17. Appendix B: four "blue curve" references renamed "extreme curve"
    (standalone supplement has no figure); "from the origin to its right
    edge" replaced by explicit beta -> 0 and beta -> infinity endpoints.
18. Appendix B: triple \citeauthor{ramsauer2021hopfield}'s construction
    replaced by one \citet plus "this line" / "Their capacity".
19. "were taken to confirm it" -> "appeared to confirm it".
20. Abstract and Discussion: undefined "live patterns" -> "drops the patterns
    entirely" / "generates no patterns".
21. Discussion: Full/Cone reach "N ~ 25 / N ~ 30" qualified with "at the
    high-alpha end".
22. Hedge words dropped: "essentially recover", "easily reach";
    "having 0.623" -> "at beta = 1 it is 0.623"; "the R+21's" -> "the R+21".

## Verification

- All three targets compile with 0 errors, no undefined references or
  citations (joint 9 pp, na 7 pp, appendices 2 pp).
- Equation numbering unchanged: main text ends at (35) in both compiles;
  joint appendix numbers land on the frozen tags (36)-(53).

## Left untouched

- Appendix C in LSE_capacity.tex still sits after \end{document} and is not
  compiled.
