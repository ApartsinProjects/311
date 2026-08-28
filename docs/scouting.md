# 311 Citizen Service Request Classification — Scouting Report

_Compiled 2026-08-28. Background research for building a 311 message classifier._

## TL;DR — Ready-to-use models with weights?

**No drop-in 311 classifier exists.** There is no published Hugging Face model,
no Kaggle checkpoint, and no GitHub release with weights specifically trained on
311 taxonomies. You need to fine-tune, or use an LLM zero/few-shot.

| Option | Weights? | Fit |
|---|---|---|
| Any HF model tagged `311` / `civic` / `service request` | ❌ none | — |
| CFPB consumer-complaint fine-tunes on HF (e.g. `Dragneel/ticket-classification-v1` DistilBERT) | ✅ | Adjacent domain, usable as warm start |
| Generic ticket-routing HF models (trained on `Tobi-Bueck/customer-support-tickets`, IT-helpdesk) | ✅ | Even further from municipal, similar short-text routing shape |
| General base LMs — `distilbert-base-uncased`, `roberta-base`, `deberta-v3-base`, `e5-large-v2`, `BGE-large`, `all-MiniLM-L6-v2` | ✅ | Realistic starting point: fine-tune or embed+kNN |
| Zero-shot LLMs (Claude, GPT-4o) | ✅ (API) | MDPI 2025 transport-complaint benchmark measured Claude ≈ 89.7%, GPT-4o ≈ 89.0%, GPT-3.5 ≈ 66.9% on a very similar civic-complaint task |
| Commercial "AI 311" CRMs (OpenGov, CivicPlus/SeeClickFix, Accela) | ❌ closed | Not usable as weights |

**Recommendation:** two-track baseline — (1) Claude/GPT-4o zero-shot as the
ceiling and (2) DistilBERT or DeBERTa-v3 fine-tuned on NYC 311 as the
deployable model. For a warmer start than random init, initialize from a
CFPB-complaint DistilBERT before fine-tuning on 311.

---

## 1. Public 311 Datasets

### 1.1 United States

| City | Portal / URL | Volume | Time coverage | Notable text/label fields | License |
|---|---|---|---|---|---|
| **New York City** | [`data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2020-to-Present/erm2-nwe9`](https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2020-to-Present/erm2-nwe9) plus [2010-2019 historical](https://catalog.data.gov/dataset/311-service-requests-from-2010-to-2019) | **~40M rows** (as of Dec 2025) across the two datasets; ~32 columns | 2010 → present, daily updates | `Complaint Type` (~450 unique), `Descriptor` (~800 unique), `Resolution Description` (free text), `Location Type`, `Agency`, `Agency Name` | NYC Open Data terms; effectively public / reuse permitted with attribution |
| **Chicago** | [`data.cityofchicago.org/Service-Requests/311-Service-Requests/v6vf-nfxy`](https://data.cityofchicago.org/Service-Requests/311-Service-Requests/v6vf-nfxy). Companion [Request Types dataset (`dgc7-2pdf`)](https://data.cityofchicago.org/Service-Requests/311-Service-Requests-Request-Types/dgc7-2pdf) | Millions of rows; **91 published request types** as of 2019, growing | New 311 CRM launched 12/18/2018; some legacy pre-2018 records flagged `LEGACY_RECORD` | `SR_TYPE`, `SR_SHORT_CODE`, `OWNER_DEPARTMENT` | Open, City of Chicago terms |
| **Boston** | [`data.boston.gov/dataset/311-service-requests`](https://data.boston.gov/dataset/311-service-requests) (Analyze Boston); per-year resources (e.g. 2024, 2025). CRM Value Codex is a downloadable PDF at the same dataset page. | ~250-300k/year (e.g. 273,951 rows in 2021); **~150 case types** | 2011 → present; backend system change Oct 2025 (some 2025 requests split between old/new tables) | `case_title`, `subject`, `reason`, `type`, `queue`, `department`, `neighborhood`, `source`, `closure_reason` | Open, City of Boston terms |
| **San Francisco** | [`data.sfgov.org/City-Infrastructure/311-Cases/vw6y-z8j6`](https://data.sfgov.org/City-Infrastructure/311-Cases/vw6y-z8j6) | Several million; 191k views / 68k downloads | 2008-07-01 → present, nightly | `Category`, `Request Type`, `Request Details`, `Responsible Agency`, `Media URL`, `Source` (channel) | Open, DataSF |
| **Los Angeles (MyLA311)** | Per-year datasets on `data.lacity.org` — e.g. [2024](https://data.lacity.org/City-Infrastructure-Service-Requests/MyLA311-Service-Request-Data-2024/b7dx-7gc3), [2025](https://data.lacity.org/City-Infrastructure-Service-Requests/MyLA311-Service-Request-Data-2025/h73f-gn57), 2015 → 2025 | ~1M/year | Aug 2015 → present | `RequestType`, `RequestSource`, `CDNumber`, `AssignTo`, `Owner`, `Address` | Open, City of LA |
| **Washington DC** | Per-year datasets on Open Data DC — e.g. [2025](https://opendata.dc.gov/datasets/85a2474b468b4dfeb84f9165210fe5a3_18) and back to 2009 | Millions across years | 2009 → present, hourly refresh on "last 30 days" view | `SERVICECODEDESCRIPTION`, `SERVICETYPECODEDESCRIPTION`, `ORGANIZATIONACRONYM`, `INSPECTIONFLAG` | Open |
| **Philadelphia (Philly311)** | [`opendataphilly.org/datasets/311-service-and-information-requests/`](https://opendataphilly.org/datasets/311-service-and-information-requests/); Carto-backed API | Millions | 2014-12-08 → present | `service_name`, `service_code`, `agency_responsible`, `description` (short), `requested_datetime` | Open |
| **Seattle** | [`data.seattle.gov/City-Administration/Customer-Service-Requests/5ngg-rpne`](https://data.seattle.gov/City-Administration/Customer-Service-Requests/5ngg-rpne) — the "Find It, Fix It" system | Hundreds of thousands | 2017 → present (some going back to 2013), quarterly refresh | `Request Type`, `Department`, `Neighborhood` | Open |
| **Kansas City, MO** | 311 KCMO on `data.kcmo.org` | 20M+ requests referenced in Kontokosta 2021 | 2007 → present | `CASE TYPE`, `WORK GROUP`, `DEPARTMENT`, `SOURCE` | Open |
| **Syracuse, NY** | SeeClickFix-powered on city portal | Smaller (city size) | 2011 → present | `Request Type`, `Category` | Open |

Many mid-size cities publish through **SeeClickFix / Open311** endpoints (below).

### 1.2 International / Canada

- **Toronto** — [`open.toronto.ca`](https://open.toronto.ca/) publishes "311 Service Requests - Customer Initiated" as one ZIP per calendar year (`SR{YEAR}.zip`); updated monthly. Coverage advertised Jan 2024 → present with ~1.1M records on file at the time of writing; earlier years also available. Fields include `Service Request Type`, `Division`, `Section`, `Ward`.
- **Montreal** — 311 request logs published on the Montreal open data portal (`donnees.montreal.ca`); French-language `TYPE_TACHE` and `NATURE`. Multilingual angle is unique.
- **Vancouver** — 311 data since 2009 published on `opendata.vancouver.ca`, aggregated by 22 planning areas.
- **UK — FixMyStreet** — [`data.mysociety.org/categories/fixmystreet/`](https://data.mysociety.org/categories/fixmystreet/). Not government-run but the de-facto UK equivalent. Aggregate counts by LSOA/LAD publicly downloadable; coordinate-level reports available for research use. 11-year longitudinal study exists.
- **London Datastore** — [`data.london.gov.uk`](https://data.london.gov.uk/) hosts borough-level complaint datasets.

### 1.3 Aggregators / cross-city

- **[311info.com](https://311info.com/methodology/categories/)** — normalizes ~24 US/Canadian cities into cross-city categories using an LLM-assisted mapping; publishes the mapping table. Handy prior art for taxonomy reconciliation.
- **[Andrew Friedman "National 311 Data Portal"](https://andrew-friedman.github.io/jkan/)** — jkan static catalog aggregating city 311 datasets.
- **[Open311 GeoReport v2](https://wiki.open311.org/GeoReport_v2/)** — the API standard. `service_code` is defined per-jurisdiction; there is **no cross-city standard taxonomy**.
- **SeeClickFix Open311 endpoint** — [`seeclickfix.com/open311/v2/docs`](https://seeclickfix.com/open311/v2/docs). Underpins hundreds of small/mid US cities.
- **MotherDuck sample** — NYC 311 published as a [DuckDB-queryable sample](https://motherduck.com/docs/getting-started/sample-data-queries/nyc-311-data/).

### 1.4 On Kaggle & Hugging Face

- Kaggle: [`san-francisco/sf-311-cases`](https://www.kaggle.com/datasets/san-francisco/sf-311-cases), [`chicago/chicago-311-service-requests`](https://www.kaggle.com/datasets/chicago/chicago-311-service-requests), [`new-york-city/ny-311-service-requests`](https://www.kaggle.com/datasets/new-york-city/ny-311-service-requests), [`city-of-seattle/seattle-csr-public-requests`](https://www.kaggle.com/datasets/city-of-seattle/seattle-csr-public-requests), [`shubhammore12/nyc-311-customer-service-requests-analysis`](https://www.kaggle.com/datasets/shubhammore12/nyc-311-customer-service-requests-analysis).
- One prior **Kaggle competition** on 311: "[NYC 311 Service Level Agreements](https://www.kaggle.com/competitions/nyc-311-service-level-agreements-20201111/overview/description)" — predicts time-to-close, not classification.
- **Hugging Face**: no canonical 311-classification benchmark exists as of this scout. Adjacent ticket-classification sets: [`Tobi-Bueck/customer-support-tickets`](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets) (61.8k, EN+DE), [`Console-AI/IT-helpdesk-synthetic-tickets`](https://huggingface.co/datasets/Console-AI/IT-helpdesk-synthetic-tickets), [`phi-ai-info/support_tickets`](https://huggingface.co/datasets/phi-ai-info/support_tickets), [`Dragneel/ticket-classification-v1`](https://huggingface.co/Dragneel/ticket-classification-v1) (a DistilBERT model). This is a gap you could fill.

---

## 2. Taxonomies / Label Spaces

**No standard.** Each jurisdiction defines its own `service_code`/`Complaint Type`. Some observed shapes:

- **NYC — hierarchical, ~2 levels.** `Complaint Type` (~450 unique historically, ~200 active) is the coarse level. `Descriptor` (~800 unique) is the fine level. There is a many-to-many `Agency` mapping (NYPD, DOB, DSNY, HPD, DEP, DOT, DPR, TLC, DOF, DOHMH are the big responders). NYC also publishes a helper table [NYC 311 Complaint Type / Descriptor / Count](https://data.cityofnewyork.us/Social-Services/NYC-311-Complaint-Type-Descriptor-Count/dtbq-f5rx) enumerating co-occurrences.
- **Chicago — flat, ~91 request types** in the modern (Dec 2018+) CRM, each mapped to an `OWNER_DEPARTMENT`. Old system used one dataset per type (~12 types).
- **Boston — 2 levels: `reason` (broad, e.g. "Streets", "Trash and Recycling") + `type` (~150 case types).** The CRM Value Codex PDF is the authoritative list.
- **DC — 2 levels: `SERVICECODE` (broad) / `SERVICETYPECODEDESCRIPTION` (narrow)** with ~200 fine codes.
- **Philadelphia — flat, ~100+ `service_name` values.**
- **LA — flat MyLA311 `RequestType`** with ~15-20 high-level categories (bulky items, dead animal, e-waste, graffiti, homeless encampment, illegal dumping, metal/household, streetlight, pothole, weed abatement, etc.).
- **Seattle — flat `Request Type`** aligned to responsible department.

Common conceptual cluster (present in almost every city): **Streets/Potholes/Sidewalks**, **Sanitation/Trash/Recycling**, **Noise**, **Graffiti**, **Illegal Parking**, **Trees/Parks**, **Streetlights**, **Rodents/Pests**, **Water/Sewer**, **Housing/Building Code**, **Homeless/Encampment**, **Animal Services**, **Abandoned Vehicle**. That common core is roughly the space 311info uses.

Salient patterns:
- **Long tail.** In NYC a handful of complaint types (Noise, Illegal Parking, HEAT/HOT WATER, Blocked Driveway, Street Condition) dominate; hundreds of types have <0.1% share.
- **Overlap / ambiguity.** e.g. NYC has both "Noise" and "Noise - Residential" and "Noise - Street/Sidewalk"; multiple descriptors of "Loud Music/Party" exist. Boston distinguishes ~6 flavors of noise.
- **Agency vs. type.** Agency assignment is deterministic in some cities and probabilistic (routed by dispatcher) in others; more than 50% of NYC 311 volume goes to NYPD, so agency prediction is severely class-imbalanced.

---

## 3. Prior Academic Work

### 3.1 Directly on 311 text/type classification

- **Hashemi et al., "Automatic Type Detection of 311 Service Requests Based on Customer Provided Descriptions"** — *Applied Artificial Intelligence*, Taylor & Francis, 2022 (Vol 36, No 1). [`doi.org/10.1080/08839514.2022.2073717`](https://www.tandfonline.com/doi/full/10.1080/08839514.2022.2073717). Standardizes categories across two cities; trains an RNN on customer descriptions; reports >83% generalization accuracy. Closest thing to a direct baseline.

- **Kontokosta, Hong, Korsberg, "Equity in 311 Reporting: Understanding Socio-Spatial Differentials in the Propensity to Complain"** — arXiv [1710.02452](https://arxiv.org/abs/1710.02452), 2017; extended version in *Sustainable Cities and Society* 64 (2021) 102503, [par.nsf.gov/servlets/purl/10314387](https://par.nsf.gov/servlets/purl/10314387). Uses gradient boosting on 20M+ NYC 311 rows to predict likelihood of HPD violations, then compares to reported complaints to quantify under-reporting.

- **Wang, Kontokosta, "Structure of 311 Service Requests as a Signature of Urban Location"** — arXiv [1611.06660](https://arxiv.org/pdf/1611.06660); *PLOS One* 2017. Uses the distribution over complaint types as a fingerprint for neighborhoods.

- **Agostini, Pierson, Garg, "A Bayesian Spatial Model to Correct Under-Reporting in Urban Crowdsourcing"** — arXiv [2312.11754](https://arxiv.org/abs/2312.11754); AAAI 2024. Applies to NYC flooding 311 reports. Code: [`github.com/gsagostini/networks_underreporting`](https://github.com/gsagostini/networks_underreporting).

- **Liu, Garg, Pierson, "Quantifying Spatial Under-reporting Disparities in Resident Crowdsourcing"** — arXiv [2204.08620](https://arxiv.org/pdf/2204.08620).

- **"Urban Incident Prediction with Graph Neural Networks: Integrating Government Ratings and Crowdsourced Reports"** — arXiv [2506.08740](https://arxiv.org/abs/2506.08740); AAAI 2026 (Proceedings vol. 41158). Uses 9.6M NYC 311 reports across 139 types + 1M government inspection ratings across 5 types (2021-2023) to train a multiview multioutput GNN.

- **"Multi-Task Anti-Causal Learning for Reconstructing Urban Events from Residents' Reports"** — arXiv [2603.11546](https://arxiv.org/abs/2603.11546), 2026. NYC 311 as the case study; MTAC framework exploiting cross-task invariances to reconstruct urban events (illegal dumping, blocked driveways, sanitary issues) from noisy resident reports.

- **"Scaling the Queue: Reinforcement Learning for Equitable Call Classification Capacity in NYC Municipal Complaint Systems"** — arXiv [2605.06482](https://arxiv.org/pdf/2605.06482), 2026. NYC Department of Buildings; RL agent triaging incoming complaints into {escalate, batch, defer, inspect now} with equity constraints.

- **Xu et al., "Determinants of citizen-generated data in a smart city: Analysis of 311 system user behavior"** — *Sustainable Cities and Society* 2020. [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S2210670720301542). LDA-style analysis.

- **Estimating reporting bias in 311 complaint data** — *Annals of Applied Statistics* 19(2), 2025, DOI [10.1214/24-AOAS2003](https://projecteuclid.org/journals/annals-of-applied-statistics/volume-19/issue-2/Estimating-reporting-bias-in-311-complaint-data/10.1214/24-AOAS2003.short). Marron Institute (NYU) team.

- **Wang, Kontokosta, "Bias in smart city governance: How socio-spatial disparities in 311 complaint behavior impact the fairness of data-driven decisions"** — *Sustainable Cities and Society* 2020. [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S2210670720307216).

- **"Predicting demand for 311 non-emergency municipal services: An adaptive space-time kernel approach"** — *Cities* 2018. [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0143622817304538). Inhomogeneous Poisson process for demand forecasting.

- **Minkoff, "NYC 311: A Tract-Level Analysis of Citizen-Government Contacting"** — *Urban Affairs Review*, SAGE 2016. [journals.sagepub.com/doi/abs/10.1177/1078087415577796](https://journals.sagepub.com/doi/abs/10.1177/1078087415577796). Political-science angle.

- **"Back to Basics: City Services and 311 Service Requests"** — ResearchGate publication 356889999, 2021. Cross-city comparison framing.

- **Data(-)based ambivalence regarding NYC 311 data infrastructure** — *Cultural Studies* 35(4-5), 2021. [tandfonline.com/doi/abs/10.1080/09502386.2021.1895256](https://www.tandfonline.com/doi/abs/10.1080/09502386.2021.1895256). Qualitative critique of the pipeline.

- **Schiff & Schiff, "Does collective citizen input impact government service provision? Evidence from SeeClickFix requests"** — *Public Administration Review*, Wiley 2025. [doi.org/10.1111/puar.13747](https://onlinelibrary.wiley.com/doi/10.1111/puar.13747). Panel dataset of ~70k requests across 100 SeeClickFix cities.

- **"Infrastructure legibility — a comparative analysis of open311-based citizen feedback systems"** — comparative study of SeeClickFix vs. Citizens Connect (Boston) engagement patterns.

- **CitySolution: A complaining task distributive mobile application for smart city corporation using deep learning** — arXiv [2410.12882](https://arxiv.org/abs/2410.12882), 2024. Image-based (not text), 4-class dataset (Damaged Road, Flood, Trash, Homeless People) of 5,494 images. Bangladesh context. Included as a "civic complaint classification" data point.

### 3.2 Adjacent civic / grievance NLP

- **"A zero-shot LLM framework for multimodal grievance classification, urgency scoring, and abuse detection in civic feedback systems"** — *Scientific Reports* (Nature) 2025, [s41598-025-32079-7](https://www.nature.com/articles/s41598-025-32079-7). Microservice architecture, zero-shot semantic routing, sentiment-derived urgency, abuse detection.

- **"Automated Classification of Public Transport Complaints via Text Mining Using LLMs and Embeddings"** — *Information* (MDPI) 16(8):644, 2025, [doi.org/10.3390/info16080644](https://doi.org/10.3390/info16080644). Compares Claude, GPT-4o, GPT-3.5 and instruction-tuned embedding models. **Reported accuracies: Claude 89.68%, GPT-4o 89.00%, GPT-3.5 66.92%.** Directly relevant few-shot benchmark methodology.

- **"Global Embeddings, Local Signals: Zero-Shot Sentiment Analysis of Transport Complaints"** — *Informatics* (MDPI) 12(3):82, 2025. 2,400-complaint multilingual corpus, "one encoder any facet" framework.

- **"Topic-sentiment analysis of citizen environmental complaints in China: Using a Stacking-BERT model"** — *Journal of Environmental Management* 2024, [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0301479724030986). 102,782 environmental complaints (2016-2022).

- **"Analyzing public demands on China's online government inquiry platform: A BERTopic-Based topic modeling study"** — *PLOS One* 2024, [journals.plos.org/plosone/article?id=10.1371/journal.pone.0296855](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0296855). SBERT+UMAP+HDBSCAN+c-TF-IDF; 104 auto-discovered topics.

- **"Proactive Complaint Management in Public Sector Informatics Using AI: A Semantic Pattern Recognition Framework"** — *Applied Sciences* 15(12):6673, 2025.

- **"Explainable Multilingual Civic Complaint Resolution System"** — *IJRASET*. MuRIL embeddings + structured urgency prediction.

- **CivicSense: AI-Based Citizen Complaint Analyzer** — Atlantis Press proceedings.

- **"Design of Emergency Call Record Support System Applying Natural Language Processing Techniques"** — Springer 2019. ASR → NER → TF-IDF/SVM call classification. Adjacent (911, not 311).

- **"Deep multitask ensemble classification of emergency medical call incidents combining multimodal data"** — medRxiv 2020.

- **"The Feasibility of Using Machine Learning to Classify Calls to South African Emergency Dispatch Centres"** — PMC 8472370.

- **"Research on automatic labeling of imbalanced texts of customer complaints based on text enhancement and layer-by-layer semantic matching"** — PMC 8178387. BERT+word2vec+text augmentation for hierarchical imbalanced customer complaints.

- **"Medical chief complaint classification with hierarchical structure of label descriptions"** — *Expert Systems with Applications* 2024. Hierarchical label description use.

- **"Recent Advances in Hierarchical Multi-label Text Classification: A Survey"** — arXiv [2307.16265](https://arxiv.org/pdf/2307.16265). Methods survey directly applicable to NYC's Type/Descriptor hierarchy.

### 3.3 CFPB consumer complaint work (methodologically transferable)

- **From Complaint Narratives to Monetary Relief: A Hybrid Machine Learning Framework for CFPB Consumer Complaints** — arXiv [2606.22664](https://arxiv.org/pdf/2606.22664), 2026. AUC-ROC 0.78 on relief prediction.
- **CFPB Consumer Complaints Analysis Using Hadoop** — arXiv [2310.06076](https://arxiv.org/pdf/2310.06076), 2023.
- **Predictive Analysis of CFPB Consumer Complaints Using Machine Learning** — arXiv [2407.06399](https://arxiv.org/pdf/2407.06399), 2024.
- **NLP-Based Consumer Complaint Assessment** — *Applied Sciences* 2025, [doi.org/10.3390/app16125992](https://doi.org/10.3390/app16125992). BERT reported accuracy ~0.72-0.79 on product classification.
- **Think Before You Classify: The Rise of Reasoning LLMs for Consumer Complaint Detection and Classification** — *Electronics* (MDPI) 14(6):1070, 2025.

---

## 4. Prior Industry / Government / Open-Source Work

- **Boston's ML routing pilot** — the city crowdsources short user descriptions on a training-data site so that a plain-language sentence ("My street has a hole in it") can be routed to the pothole intake form. Search-engine-style model. Coverage: [StateScoop](https://statescoop.com/soon-machine-learning-will-make-it-easier-to-submit-311-requests-in-boston/), [Boston.gov "Help us teach the new 311"](https://www.boston.gov/news/help-us-teach-new-311). In late 2024/2025 Boston selected an "AI-native" 311 platform to replace its legacy CRM ([GovTech](https://www.govtech.com/artificial-intelligence/boston-311-services-shaped-by-adaptable-ai-powered-platform)).
- **OpenGov 311** — commercial AI-native 311 CRM ([`opengov.com/products/government-app-library/311-request-management/`](https://opengov.com/products/government-app-library/311-request-management/)). Advertises intake completeness checks, location validation, duplicate detection, auto-categorization.
- **CivicPlus SeeClickFix 311 CRM** — [`civicplus.com/seeclickfix-311-crm/`](https://www.civicplus.com/seeclickfix-311-crm/).
- **Accela / Catalis / NebuLogic** — other municipal 311 SaaS with published claims about NLP-based intake routing.
- **NYC DOB Data-Driven Enforcement** — the equity-centered RL routing framework covered in arXiv 2605.06482 is described by NYC as production-adjacent.
- **Harvard Kennedy School Data-Smart City Solutions** — [`datasmart.hks.harvard.edu/news/article/cities-embrace-new-improved-311-services`](https://datasmart.hks.harvard.edu/news/article/cities-embrace-new-improved-311-services). Overview of city 311 modernization projects.
- **GitHub — general purpose exploratory notebooks (dozens exist)**:
  - [`doshiharmish/Boston-311-Service-Requests-Analysis-2021`](https://github.com/doshiharmish/Boston-311-Service-Requests-Analysis-2021)
  - [`mrkeville/SeeClickFix-Analysis-Syracuse-NY`](https://github.com/mrkeville/SeeClickFix-Analysis-Syracuse-NY)
  - [`AvonleaFisher/Analyzing-NYC-311-Service-Requests`](https://github.com/AvonleaFisher/Analyzing-NYC-311-Service-Requests)
  - `github.com/topics/open311` (many small implementations)
  - `github.com/SeeClickFix/Open311` — reference implementation of Open311 API.
- **Blog posts / capstones** (useful as informal baselines):
  - [Towards Data Science — "Analyzing and Modelling NYC 311 Service Requests"](https://towardsdatascience.com/analyzing-and-modelling-nyc-311-service-requests-eb6a9c9adc7c/) — text→agency classifier, notes NYPD-heavy majority baseline of ~50%.
  - [Bopardikar — "Using ML to Predict LA 311 ETC"](https://medium.com/@vbopardi_49658/using-machine-learning-to-predict-los-angeles-311-service-requests-estimated-time-of-completion-5829223679e4) — random forest on completion time.
  - NYC Data Science blog — "Detailed Data Analysis: The Rise of NYC 311 Noise Complaints".
- **Kaggle**: no public 311-**classification** benchmark, only exploratory notebooks and an SLA-prediction competition.
- **Hugging Face**: no published 311 text-classification model or dataset.

---

## 5. Related Adjacent Datasets

| Dataset | Why relevant | Size | Access |
|---|---|---|---|
| **CFPB Consumer Complaint Database** | Product/sub-product/issue text classification with a narrative; canonical benchmark for complaint NLP | ~4M+ complaints (~13.8M+ if counting company responses) | [`consumerfinance.gov/data-research/consumer-complaints/`](https://www.consumerfinance.gov/data-research/consumer-complaints/); public API |
| **Customer Support Tickets** (`Tobi-Bueck/customer-support-tickets`) | Text→queue routing, EN+DE | 61.8k | HF |
| **IT Helpdesk Synthetic Tickets** (Console-AI) | Balanced synthetic training set for ticket routing | ~thousands | HF |
| **Kaggle "Customer Support Ticket Dataset"** (suraj520) | General ticket classification | ~10k | Kaggle |
| **Seattle 911 CAD dispatch** (`data.seattle.gov`) and other municipal CAD dumps | 911-side of civic ML; short text + event type | Millions | Open |
| **Reddit r/legaladvice / grievance corpora** | Long-tail informal complaint text | Millions | HF, Pushshift |
| **FEMA Disaster reports, US National Highway ARF, USDA Complaint text** | Short-narrative government text | Varies | data.gov |
| **20 Newsgroups, DBpedia, AG News** | Standard text-classification baselines for pretraining ablations | Small | HF/UCI |
| **HiClass** — [`arxiv.org/pdf/2112.06560`](https://arxiv.org/pdf/2112.06560) | scikit-learn-compatible hierarchical classifier library; useful if you exploit NYC Type→Descriptor hierarchy | — | GitHub |
| **BERTopic, contextualized topic models** | For unsupervised discovery when labels drift | — | HF/GitHub |

---

## 6. Key Challenges Documented in Prior Work

1. **Severe class imbalance.** In NYC ~50%+ of requests route to NYPD (mostly noise + illegal parking). A "predict majority agency" baseline is already ~50%. Long tail of low-frequency complaint types under-fit (Hashemi 2022; multiple TDS blog posts; the Chinese customer-complaint text-enhancement paper).

2. **Taxonomy drift and inconsistency.**
   - Cities re-organize categories periodically (Chicago's 2018-12-18 CRM rewrite; NYC descriptors added/renamed; Boston's Oct-2025 backend transition splitting the dataset).
   - `Complaint Type` / `Descriptor` co-occurrence is many-to-many; single free-text description often maps to multiple valid labels.
   - No cross-city standard (Open311 leaves `service_code` to the jurisdiction), so multi-city training requires an explicit label reconciliation step — cf. 311info's LLM-assisted mapping and Hashemi 2022's "standardize categories across two cities" step.

3. **Free-text ambiguity and short-text sparsity.** Boston's short titles and Philly's `description` fields are often <15 tokens; customer descriptions may say "hole in street" (pothole? sinkhole? construction defect?). The MDPI 2025 LLM-vs-embedding paper found Claude ~89.7% but GPT-3.5 ~67% — model choice matters heavily on short civic text.

4. **Reporting/geographic bias (fairness).** Kontokosta et al. 2017 and 2021, Wang & Kontokosta 2020, Agostini/Pierson/Garg AAAI 2024, Marron/AoAS 2025 — all document that low-English-proficiency, high-unemployment, non-white, and lower-income areas under-report. Training on raw counts therefore *learns* the bias. This affects downstream classification whenever priors depend on location.

5. **Missing-not-at-random outcome labels.** Enforcement/inspection outcomes are systematically less recorded in historically underserved areas, so any supervised training on `Resolution Description` or SLA-met labels inherits that missingness.

6. **Multi-jurisdictional generalization.** Same phrase ("blocked bike lane") routes to different owning agencies in NYC, Chicago, LA. Fine-tuning per-city vs. one shared model is an open trade-off.

7. **Channel bias.** Text quality varies drastically by intake channel — call-center dispatcher transcription vs. mobile-app self-report vs. Twitter/X vs. email. San Francisco survival-analysis work found Open311 vs. Twitter reports have different resolution-time distributions; the same is likely true for text style.

8. **Duplicates / near-duplicates.** Same event reported many times; deduplication is a documented commercial-CRM feature (OpenGov, Accela).

9. **Multilingual and code-switching.** Real intake includes Spanish, Chinese, Bengali, French (Montreal), Russian. Most public research has been English-only.

10. **Label-vocabulary evolution over time.** A 10-year classifier must handle types that didn't exist earlier (e.g. "e-scooter", "encampment"), i.e. concept-drift.

11. **Aleatoric vs. epistemic uncertainty in tiny categories.** ~800 NYC descriptors mean many have <100 training examples; hierarchical multi-label methods (see arXiv 2307.16265 survey; HiClass; Hierarchical MixUp arXiv 2209.13912) become relevant.

12. **Data-infrastructure critique.** Cao 2021 (*Cultural Studies*) argues the 311 pipeline itself is not neutral — worth reading if the classifier is exposed to end users.

---

## 7. Suggested Starting Points for Building the Classifier

Given the above scout, the highest-leverage starting configuration is:

- **Anchor dataset**: NYC 311 (largest, richest text, well-documented hierarchy) → union with Chicago + Boston for cross-city generalization tests. Use 311info's mapping as a starting-point normalized taxonomy.
- **Text field**: concatenate `Descriptor` + `Location Type` + any `Resolution Description` for NYC; `case_title` + `subject` + `reason` for Boston; `service_name` + `description` for Philly.
- **Task variants worth benchmarking**:
  1. Coarse `Complaint Type` classification (~150 classes) — closest to Hashemi 2022 baseline (~83%).
  2. Fine `Descriptor` classification (~800 classes) — hierarchical, per HiClass / arXiv 2307.16265.
  3. `Agency` classification (~10-20 classes) — matches Boston's routing pilot.
  4. Zero/few-shot LLM baseline — reproduce the MDPI 2025 methodology (Claude vs. GPT-4o vs. embedding+kNN).
- **Fairness sanity checks**: stratify metrics by ZIP/tract using the Kontokosta / Agostini / Marron-Institute methodology.
- **Baselines to include**: TF-IDF + LinearSVC/LogReg (classic 311 baseline), DistilBERT/BERT fine-tuning, sentence-transformer embeddings + kNN, an LLM few-shot classifier, plus a hierarchical variant.

---

## Sources (main URLs cited)

**Datasets:** [NYC 2020-present](https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2020-to-Present/erm2-nwe9), [NYC 2010-2019](https://catalog.data.gov/dataset/311-service-requests-from-2010-to-2019), [Chicago v6vf-nfxy](https://data.cityofchicago.org/Service-Requests/311-Service-Requests/v6vf-nfxy), [Chicago Request Types](https://data.cityofchicago.org/Service-Requests/311-Service-Requests-Request-Types/dgc7-2pdf), [Boston Analyze Boston](https://data.boston.gov/dataset/311-service-requests), [SF vw6y-z8j6](https://data.sfgov.org/City-Infrastructure/311-Cases/vw6y-z8j6), [LA 2025](https://data.lacity.org/City-Infrastructure-Service-Requests/MyLA311-Service-Request-Data-2025/h73f-gn57), [DC 2025](https://opendata.dc.gov/datasets/85a2474b468b4dfeb84f9165210fe5a3_18), [Philly](https://opendataphilly.org/datasets/311-service-and-information-requests/), [Seattle 5ngg-rpne](https://data.seattle.gov/City-Administration/Customer-Service-Requests/5ngg-rpne), [Toronto](https://open.toronto.ca/), [FixMyStreet](https://data.mysociety.org/categories/fixmystreet/), [311info categories](https://311info.com/methodology/categories/), [Open311 GeoReport v2](https://wiki.open311.org/GeoReport_v2/), [SeeClickFix Open311 docs](https://seeclickfix.com/open311/v2/docs).

**Papers:** [Hashemi 2022 (T&F)](https://www.tandfonline.com/doi/full/10.1080/08839514.2022.2073717), [Kontokosta 2017 arXiv](https://arxiv.org/abs/1710.02452), [Wang & Kontokosta 2017 PLOS One / arXiv 1611.06660](https://arxiv.org/pdf/1611.06660), [Agostini/Pierson/Garg 2024 AAAI arXiv 2312.11754](https://arxiv.org/abs/2312.11754), [code repo](https://github.com/gsagostini/networks_underreporting), [Liu et al. 2204.08620](https://arxiv.org/pdf/2204.08620), [GNN urban incident 2506.08740](https://arxiv.org/abs/2506.08740), [MTAC 2603.11546](https://arxiv.org/abs/2603.11546), [RL for NYC DOB 2605.06482](https://arxiv.org/pdf/2605.06482), [Xu Sustainable Cities 2020](https://www.sciencedirect.com/science/article/abs/pii/S2210670720301542), [Wang & Kontokosta 2020 Sustainable Cities](https://www.sciencedirect.com/science/article/abs/pii/S2210670720307216), [AoAS 2025 estimating reporting bias](https://projecteuclid.org/journals/annals-of-applied-statistics/volume-19/issue-2/Estimating-reporting-bias-in-311-complaint-data/10.1214/24-AOAS2003.short), [Zero-shot LLM civic Nature Sci Rep 2025](https://www.nature.com/articles/s41598-025-32079-7), [MDPI Info 2025 transport LLM](https://doi.org/10.3390/info16080644), [BERTopic China gov PLOS 2024](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0296855), [Stacking-BERT China env complaints](https://www.sciencedirect.com/science/article/abs/pii/S0301479724030986), [CitySolution 2410.12882](https://arxiv.org/abs/2410.12882), [Hierarchical multi-label survey 2307.16265](https://arxiv.org/pdf/2307.16265), [HiClass 2112.06560](https://arxiv.org/pdf/2112.06560), [SeeClickFix panel Schiff PAR 2025](https://onlinelibrary.wiley.com/doi/10.1111/puar.13747), [Kontokosta 2021 Sust. Cities PDF](https://par.nsf.gov/servlets/purl/10314387).

**CFPB:** [Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/), [arXiv 2606.22664](https://arxiv.org/pdf/2606.22664), [arXiv 2407.06399](https://arxiv.org/pdf/2407.06399), [arXiv 2310.06076](https://arxiv.org/pdf/2310.06076).

**Industry / gov:** [Boston ML routing StateScoop](https://statescoop.com/soon-machine-learning-will-make-it-easier-to-submit-311-requests-in-boston/), [Boston "Help us teach the new 311"](https://www.boston.gov/news/help-us-teach-new-311), [OpenGov 311](https://opengov.com/products/government-app-library/311-request-management/), [CivicPlus SeeClickFix](https://www.civicplus.com/seeclickfix-311-crm/), [Harvard Data-Smart Cities 311 overview](https://datasmart.hks.harvard.edu/news/article/cities-embrace-new-improved-311-services).
