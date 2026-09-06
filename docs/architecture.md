# Architecture

Detail behind the overview in the root [`README.md`](../README.md). Everything
here is derived from the notebook code and from the project presentations in
[`presentations/`](presentations/); where a fact comes from the presentations
rather than the code, it is marked as such.

## Design intent

The project addresses four problems stated in
`presentations/2024_prm_recommender_esprit.pptx`:

- PRM terminology is not standardised across practitioners.
- There is no cohesive body of knowledge with a shared conceptual model.
- Competing PM best practices, standards and guidelines overlap.
- PM framework language is not computer-interpretable.

The response is ontology learning: convert a human-readable standard into a
formal, machine-interpretable conceptual graph, then reason and recommend over
it. PMI's standards were chosen as the corpus because they standardise generally
accepted PM concepts and supply a comprehensive set of definitions.

## Module 1 — Knowledge retrieval

**Goal:** extract concepts, instances and relationships from PMI's PDFs.

**Stages**

1. **Raw text extraction.** Three different libraries are used across the
   notebooks, each for a different reason:
   - `PyPDF2` (notebook 01) — whole-document text, simple and fast.
   - `pdfplumber` (notebook 03) — page-range extraction with a character filter
     that drops glyphs larger than 30pt, removing headings from the body text:
     `page.filter(lambda obj: not (obj["object_type"] == "char" and obj["size"] > 30))`
   - `PyMuPDF` / `fitz` (notebook 03) — title and description extraction driven
     by numbered-section regexes, with BERT masked-language-model completion of
     truncated descriptions.
2. **Structural cleanup.** `remove_section()` strips the table of contents and
   list of figures by keyword span; `remove_footers()` / `remove_headers()`
   delete the PMI copyright line and chapter headers by regex.
3. **Ligature repair.** `correct_extraction_mistakes()` fixes damage that PDF
   extraction inflicts on this specific typesetting — `de nes` → `defines`,
   `speciﬁ c` → `specific`, `identiﬁ ed` → `identified`, and dozens more. This
   is corpus-specific and must be kept.
4. **Sectioning by process.** `extract_text(text, startT, endT)` slices the
   document between section markers, producing one text block per PMI process:

   | Process | Marker span |
   |---|---|
   | Introduction | `1.1` → `2.1` |
   | Principles and concepts | `2.1` → `3.1` |
   | Introduction to PRM processes | `3.1` → `4.1` |
   | Plan Risk Management | `4.1` → `5.1` |
   | Identify Risks | `5.1` → `6.1` |
   | Perform Qualitative Risk Analysis | `6.1` → `7.1` |
   | Perform Quantitative Risk Analysis | `7.1` → `8.1` |
   | Plan Risk Responses | `8.1` → `9.1` |
   | Monitor and Control Risks | `9.1` → `A.1` |

   This makes the pipeline **numbering-sensitive**: a different edition of the
   standard renumbers the sections and the slicing silently produces wrong or
   empty blocks.
5. **Linguistic processing.** Sentence tokenisation (NLTK), POS-aware
   lemmatisation, stopword removal, spell correction, and spaCy NER/POS tagging.
6. **Relation extraction**, by three independent mechanisms:
   - `extract_grammatical_relations()` — dependency labels `nsubj`, `dobj`,
     `prep`, `pobj`.
   - `extract_semantic_relations()` — verb-centred SVO extraction over `ROOT`,
     `acl`, `relcl`, `xcomp`, `ccomp`, with an exclusion list for pronouns and
     determiners (`which`, `it`, `that`, `these`, …).
   - **REBEL** (`Babelscape/rebel-large`, notebook 03) — neural end-to-end
     triplet generation, wrapped in a `KB` class that deduplicates relations.
7. **Definition generation.** GPT-2 (`gpt2-medium`) is fine-tuned on the
   extracted risk corpus, then prompted per concept (`"define the [concept]
   concept."`) to produce domain-consistent definitions.

## Module 2 — Conceptual graph

**Entity organisation** (notebook 02). Extracted entities are sorted into OWL
constructs by rule:

- Concepts and individuals — from named entities and spaCy `Matcher` patterns.
- Concepts and sub-concepts — from the "is a" rule.
- Properties — from the subject-predicate-object rule:
  - `A Predicate B` where both A and B are individuals → **object property**
  - `A Predicate B` where A is an individual and B is not → **data property**

The ontology is materialised with `owlready2` as `Ontology_PMBOKV2.0.owl`:
classes and subclasses, data properties, object properties, individuals, then
per-process annotations covering **inputs, tools and techniques, and outputs**.

Reported in the presentation (not verified from a committed artifact): 1,492
concepts, 360 object properties, 66 instances, 90 data properties, 187 axioms
and 22 SWRL rules.

**Graph representation.** In parallel, the same triples become:

- an `rdflib.Graph` of `(subject, predicate, object)` URIs, and
- a `networkx` graph — `DiGraph` in notebooks 02/04, and
  `nx.from_pandas_edgelist(df, 'Concepts', 'Concept_of_type_relation', edge_attr='Type_relation')`
  in notebook 05.

**Embeddings.** Two independent strategies, both retained:

| | Notebook 04 | Notebook 05 |
|---|---|---|
| Node embedding | TransE via PyKEEN, dim 64 | node2vec, dim 64, walk length 30, 200 walks |
| Text features | — | sentence-transformers `paraphrase-MiniLM-L6-v2` (384-d) over `Definition` and `Process_name` |
| Edge embedding | — | Hadamard / average / weighted-L1 / **weighted-L2** (weighted-L2 is the one used) |
| Split | 80 / 10 / 10, `random_state=42` | — |
| Evaluation | PyKEEN `RankBasedEvaluator`, filtered | — |
| GNN | `DeepReasoningGNN`: input `Linear`, 4 × `GCNConv` (hidden 128), dropout 0.5, three heads (definition, synonym, relation) | 2 × `GCNConv` (384 → 384 → 153) |
| Exports | `graph_data.graphml`, `deep_reasoning_gnn_model.pth` | `node2vec.model` |

Notebook 04 additionally projects entity embeddings to 2-D with t-SNE for
visualisation, and notebook 05 renders an interactive 3-D Plotly graph using a
`nx.spring_layout(G, dim=3, seed=42)` layout with definition/synonym/process
hover labels.

## Module 3 — Recommendation

Implemented in notebook 02 as ontology traversal, not as a service:

- `Recommande(query)` — keyword-driven lookup returning the matching process
  class and its subclass tree.
- `get_subclasses(Type, classe)` — pulls the inputs / tools-and-techniques /
  outputs branch for a process.
- `get_annotation()` / `get_list_annotation()` — retrieves the OWL `comment`
  annotations attached to each element.
- `extracy_figure_page()` and `get_section_page()` — resolve a `Figure X-Y` or
  `Section N.M` reference inside an annotation back to a **page number in the
  source PDF**, by parsing the standard's own list-of-figures and
  table-of-contents pages.

The presentation describes a rule base expressed in **SWRL** (example: *if a
project risk has probability 0.8–1.0 and impact rating 100, then exposure is
very high with score 100*) and a web application front end. The SWRL rules live
in the generated ontology; **the web application does not exist in this
repository.**

## Cross-cutting: the canonical schema

Every stage exchanges data through one CSV shape — see the schema table in the
root README. Two consequences worth knowing:

- **URI form.** When a concept becomes an RDF/graph URI its spaces are replaced
  with underscores (`row['Concepts'].replace(" ", "_")`), and lookups later
  reverse this. Concept strings are therefore the primary key and must stay
  stable across stages.
- **Positional node indices.** Node and edge indices are computed with
  `list(G.nodes()).index(...)` inside per-row loops. This is correct but O(n²),
  and it makes **graph insertion order load-bearing**: reorder graph
  construction without also fixing the lookups and `edge_index` silently points
  at the wrong nodes.

## What is deliberately absent

No database, no HTTP API, no message queue, no authentication, no container, no
cloud service, no CI. The system is a batch pipeline over local files; the only
network access is anonymous model downloads from the Hugging Face Hub.
