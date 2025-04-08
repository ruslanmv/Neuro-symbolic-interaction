
# Comparisons with our Model
🧠 1. Philosophical and Functional Shift

Feature | Original Rule-Agent | Neuro Symbolic Approach
--- | --- | ---
Philosophy | Rule-based symbolic reasoning aided by LLM-generated structured inputs | Full integration of symbolic ontology, logical reasoning, machine learning, and generative LLMs
Goal | Translate NL to parameters for rule services (ODM) | Ground LLM interpretation in ontologies, reason over logic, and generate explanations
Reasoning | Delegated to ODM (external decision service) | In-graph ontology reasoning via Owlready2 + reasoning + contradiction detection
Interpretability | Output mapped back to rules and decisions | Natural Language + Logical Form + Explanation

🧩 2. Ontology and Symbolic Reasoning

Category | Original Rule-Agent | Neuro Symbolic Approach
--- | --- | ---
Ontology Use | Uses precompiled .jar ruleapp (opaque symbolic logic) | OWL ontologies dynamically defined, queried, and reasoned over using owlready2
Reasoner | External: IBM ODM | Internal: HermiT via owlready2’s sync_reasoner()
Evaluation | No direct logical validation | validate truth/falsehood of logical forms in ontology with reasoned feedback
Relationships | Preset rules (.brl) | Declarative OWL object properties (CausesFailure, etc.)

🧬 3. Learning and Generalization

Category | Original Rule-Agent | Neuro Symbolic Approach
--- | --- | ---
LLM Role | Converts queries into input JSON for rules | Synthesizes statements, verifies ontological truth, rewrites wrong statements
Data Generation | Static, hardcoded examples | Automatic training set generation from ontology relationships (true & false)
Model Training | No learning model | Trains logistic regression on NL → logical form using generated training pairs

📤 4. Pipeline Architecture Comparison

Original Pipeline:

- Natural Language → Prompt Template (LLM) → Structured JSON → Rule Service → Output

Neuro Symbolic Approach Pipeline:

- Natural Language →
- ML Classification → Logical Statement →
- Ontology Reasoning (truth/falsehood) →
- Prompt to LLM with Context →
- Natural Language Answer + Logical Form + Explanation

🧪 5. Explanation and Contradiction Detection

Capability | Original Rule-Agent | Neuro Symbolic Approach
--- | --- | ---
Contradiction? | Not handled | Explicitly identifies logical contradictions via class mismatch or disjoint types
Explanations | Human-generated | Generated using parsed ontology structure
Negative Cases | Not modeled | Actively generates and trains on true + false variations

📈 6. ML Training Dataset from Ontology

The Neuro Symbolic Approach dynamically:

- Generates (subject.property(object)) logical forms
- Evaluates true/false from ontology
- Converts to natural language (verbalizer)
- Trains ML classifier to map back from NL → logic

This is a tight neuro-symbolic loop: logic generates training data → ML learns mappings → ML produces logic → logic gets validated.

🧠 7. Model Interpretation and Generalization

Feature | Original Rule-Agent | Neuro Symbolic Approach
--- | --- | ---
Handles Uncertainty | LLM default behavior | Explicit handling with reasoned rejections, logical consistency
Ontology-Driven Response | No | Yes – ontology forms the epistemic foundation
Factual Grounding | Delegated to external service | Grounded in internal ontology with reasoner

🛠 Key Implementation Innovations

- Ontology-driven training data generator
- Verbalizer functions with rich linguistic variation
- Logical consistency checking with explanations
- LLM synthesis prompt informed by ontology, truth table, evaluation result

🧩 Core Advantages of the Neuro Symbolic Approach

1. Interpretable — Logical forms, true/false evidence, explanations  
2. Trainable — Uses symbolic facts to generate ML data  
3. Robust — Detects contradictions, supports false inputs  
4. Scalable — Extendable with new OWL ontologies  
5. Unified — Neuro and symbolic layers interact bidirectionally  

