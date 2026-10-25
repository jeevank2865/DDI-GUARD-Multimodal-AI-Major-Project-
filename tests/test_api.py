import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient

import ddi_guard.api as api_module
from ddi_guard.api import app


class ApiTests(unittest.TestCase):
    def setUp(self):
        # Summary cache and provider cooldown are process-local in production;
        # reset them between tests so one test's mocked 429 cannot affect another.
        with api_module._EXPLAINABILITY_SUMMARY_LOCK:
            api_module._EXPLAINABILITY_SUMMARY_CACHE.clear()
            api_module._EXPLAINABILITY_AI_COOLDOWN_UNTIL = 0.0
        with api_module._ASSISTANT_CHAT_COOLDOWN_LOCK:
            api_module._ASSISTANT_CHAT_COOLDOWN_UNTIL = 0.0

    def test_hero_image_route_serves_only_the_existing_jpeg(self):
        client = TestClient(app)
        image_path = Path(__file__).resolve().parent / "capsule-for-drug-delivery-3d-rendering-KYC8XP.jpg"
        response = client.get("/tests/capsule-for-drug-delivery-3d-rendering-KYC8XP.jpg")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "image/jpeg")
        self.assertEqual(response.content, image_path.read_bytes())
        self.assertEqual(client.get("/tests/test_api.py").status_code, 404)

    def test_status_serves_real_ingestion_artifacts(self):
        client = TestClient(app)
        page = client.get("/")
        self.assertEqual(page.status_code, 200)
        self.assertIn("DDI-GUARD · Research Platform", page.text)
        self.assertIn("DDI Prediction", page.text)
        self.assertIn("/api/research/predict", page.text)
        self.assertIn("DailyMed label passages", page.text)
        response = client.get("/api/status")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["ddinter"]["raw_records"], 222383)
        self.assertEqual(body["ddinter"]["unique_labeled_pairs"], 130422)
        self.assertIsNotNone(body["splits"])
        self.assertEqual(body["baseline"]["status"], "completed")
        self.assertEqual(body["baseline"]["test_pair_count"], 19564)
        self.assertIn("molecular_pair_disjoint", body)
        self.assertIn("fingerprint_rf_cold_drug", body)
        self.assertFalse(body["model_approval"]["approved"])
        self.assertEqual(body["model_approval"]["failed_checks"], [])
        self.assertEqual(body["research_candidate"]["status"], "verified_academic_candidate_research_route_enabled_production_blocked")
        self.assertTrue(body["model_approval"]["research_demo_policy"]["allowed"])
        self.assertTrue(body["model_approval"]["research_demo_policy"]["research_endpoint_enabled"])
        self.assertEqual(body["prediction_service"]["http_status"], 503)
        self.assertIn("assistant", body)

    def test_prediction_fails_closed_and_validates_inputs(self):
        client = TestClient(app)
        response = client.post("/api/predict", json={"drug_a": "DDInter1", "drug_b": "DDInter1033"})
        self.assertEqual(response.status_code, 503)
        self.assertIn("production predictions are disabled by the explicit approval gate", response.json()["detail"])
        invalid = client.post("/api/predict", json={"drug_a": "", "drug_b": "Ibuprofen"})
        self.assertEqual(invalid.status_code, 422)
        unknown = client.post("/api/predict", json={"drug_a": "UnknownDrug", "drug_b": "DDInter1"})
        self.assertEqual(unknown.status_code, 404)

    def test_drug_search_and_pair_explorer_use_reviewed_records_and_label_provenance(self):
        client = TestClient(app)
        search = client.get("/api/drugs/search", params={"q": "Abacavir"})
        self.assertEqual(search.status_code, 200)
        record = search.json()["results"][0]
        self.assertEqual(record["drug_id"], "DDInter1")
        self.assertTrue(record["structure_available"])
        drug = client.get("/api/drugs/DDInter1")
        self.assertEqual(drug.status_code, 200)
        self.assertIn("@", drug.json()["smiles"])
        pair = client.get("/api/interactions/lookup", params={
            "drug_a_id": "DDInter1", "drug_b_id": "DDInter1033"})
        self.assertEqual(pair.status_code, 200)
        body = pair.json()
        self.assertEqual(body["dataset_annotation"]["severity"], "Major")
        self.assertIn("research-dataset annotation", body["dataset_annotation"]["note"])
        self.assertEqual(body["model_prediction"]["status"], "unavailable")
        self.assertEqual(body["text_evidence_status"], "available")
        passage = body["drug_text_evidence"]["DDInter1"][0]
        self.assertIn("DailyMed", passage["source_id"])
        self.assertTrue(passage["source_url"].startswith("https://dailymed.nlm.nih.gov/"))
        self.assertTrue(passage["corpus_sha256"])
        unknown = client.get("/api/interactions/lookup", params={
            "drug_a_id": "not-a-drug", "drug_b_id": "DDInter1"})
        self.assertEqual(unknown.status_code, 404)

    def test_catalog_search_accepts_verified_alias_and_pubchem_id(self):
        client = TestClient(app)
        alias = client.get("/api/drugs/search", params={"q": "136470-78-5"})
        self.assertEqual(alias.status_code, 200)
        self.assertEqual(alias.json()["results"][0]["drug_id"], "DDInter1")
        cid = client.get("/api/drugs/search", params={"q": "441300"})
        self.assertEqual(cid.status_code, 200)
        self.assertEqual(cid.json()["results"][0]["drug_id"], "DDInter1")
        self.assertNotIn("_search_terms", cid.json()["results"][0])

    def test_drug_search_only_returns_fully_verified_eligible_records(self):
        client = TestClient(app)
        eligibility = client.get("/api/drugs/eligibility")
        self.assertEqual(eligibility.status_code, 200)
        audit = eligibility.json()
        self.assertEqual(audit["catalog_count"], 1939)
        self.assertGreater(audit["eligible_count"], 0)
        self.assertEqual(sum(audit["excluded_by_primary_reason"].values()),
                         audit["catalog_count"] - audit["eligible_count"])
        self.assertIn("no_verified_3d_conformer", audit["excluded_any_reason"])

        search = client.get("/api/drugs/search", params={"limit": 50})
        body = search.json()
        self.assertEqual(body["eligible_drug_count"], audit["eligible_count"])
        self.assertEqual(body["result_count"], audit["eligible_count"])
        self.assertTrue(body["has_more"])
        for record in body["results"]:
            self.assertTrue(record["structure_available"])
            self.assertTrue(record["structure_3d_available"])
            self.assertTrue(record["prediction_eligible"])
            self.assertGreater(record["verified_ddinter_label_count"], 0)

        all_ids = []
        offset = 0
        while True:
            page = client.get("/api/drugs/search", params={"limit": 50, "offset": offset}).json()
            all_ids.extend(row["drug_id"] for row in page["results"])
            if not page["has_more"]:
                break
            offset = page["next_offset"]
        self.assertEqual(len(all_ids), audit["eligible_count"])
        self.assertEqual(len(set(all_ids)), len(all_ids))

        # Abacavir has a hash-backed identity, renderable 2D, verified source conformer, and DDInter label.
        abacavir = client.get("/api/drugs/search", params={"q": "Abacavir"}).json()
        self.assertEqual([row["drug_id"] for row in abacavir["results"]], ["DDInter1"])

    def test_prediction_rejects_drugs_outside_verified_selector(self):
        # DDInter2 has identity, 2D structure and DDInter labels but no source-verified 3D conformer.
        client = TestClient(app)
        response = client.post("/api/predict", json={"drug_a": "DDInter2", "drug_b": "DDInter1"})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["detail"]["code"], "drug_not_prediction_eligible")
        self.assertIn("no_verified_3d_conformer", response.json()["detail"]["reasons"])

    def test_explainability_uses_exact_pair_and_distinguishes_drug_level_sources(self):
        client = TestClient(app)
        # Aminosalicylic acid + Amcinonide is a verified model-supported pair with no exact DDInter label.
        with patch("ddi_guard.api._pubmed_pair_literature", return_value={
                "status": "no_results", "search_performed": True, "items": []}), \
             patch("ddi_guard.api._live_dailymed_interaction_passage", return_value={"status": "no_interaction_section"}), \
             patch("ddi_guard.api._live_openfda_pair_labels", return_value={"status": "searched", "by_drug": {}}):
            response = client.get("/api/interactions/explainability", params={
                "drug_a_id": "DDInter75", "drug_b_id": "DDInter63"})
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["pair"]["drug_a"]["drug_id"], "DDInter75")
        self.assertEqual(body["pair"]["drug_b"]["drug_id"], "DDInter63")
        self.assertEqual(body["pair_specific_evidence"]["status"], "not_found")
        self.assertIn("does not establish that the combination is safe", body["pair_specific_evidence"]["empty_state"])
        self.assertIsNone(body["pair_specific_evidence"]["record"])
        self.assertEqual(body["molecular_context"]["DDInter75"]["molecular_formula"], "C7H7NO3")
        self.assertTrue(body["molecular_context"]["DDInter75"]["structure_3d_available"])
        target_records = body["drug_target_evidence"]["DDInter75"]
        self.assertTrue(target_records)
        self.assertEqual(target_records[0]["evidence_category"], "Drug–target bioactivity background")
        self.assertIsNone(target_records[0]["activity_value"])
        self.assertTrue(target_records[0]["target_url"].startswith("https://www.ebi.ac.uk/chembl/"))
        self.assertEqual(body["external_search"]["pubmed"]["status"], "no_results")
        searches = body["external_search"]["pair_search_links"]
        self.assertEqual(len(searches), 6)
        self.assertTrue(any("pubmed.ncbi.nlm.nih.gov" in item["url"] for item in searches))
        self.assertTrue(any("europepmc.org/search" in item["url"] for item in searches))
        self.assertTrue(all("scope" in item for item in searches))

    def test_explainability_literature_falls_back_to_europe_pmc(self):
        from ddi_guard.api import _pubmed_pair_literature
        search_result = Mock()
        search_result.__enter__ = Mock(return_value=search_result)
        search_result.__exit__ = Mock(return_value=False)
        search_result.read = Mock(return_value=b'{"esearchresult":{"idlist":[]}}')
        europe_result = Mock()
        europe_result.__enter__ = Mock(return_value=europe_result)
        europe_result.__exit__ = Mock(return_value=False)
        europe_result.read = Mock(return_value=(
            '{"resultList":{"result":[{"title":"Aminosalicylic acid and amcinonide interaction review",'
            '"abstractText":"Aminosalicylic acid and amcinonide were mentioned in a review of interactions.",'
            '"pmid":"123456","firstPublicationDate":"2022-01-01"}]}}').encode())
        _pubmed_pair_literature.cache_clear()
        with patch("ddi_guard.api.urlopen", side_effect=[search_result, europe_result]):
            result = _pubmed_pair_literature("DDInter75", "DDInter63")
        self.assertEqual(result["status"], "available")
        self.assertEqual(result["provider"], "Europe PMC")
        self.assertEqual(result["items"][0]["source_name"], "Europe PMC")
        self.assertIn("not by itself proof", result["items"][0]["interpretation"])
        _pubmed_pair_literature.cache_clear()

    def test_explainability_returns_exact_ddinter_pair_record(self):
        with patch("ddi_guard.api._pubmed_pair_literature", return_value={
                "status": "no_results", "search_performed": True, "items": []}), \
             patch("ddi_guard.api._live_dailymed_interaction_passage", return_value={"status": "no_interaction_section"}), \
             patch("ddi_guard.api._live_openfda_pair_labels", return_value={"status": "searched", "by_drug": {}}):
            response = TestClient(app).get("/api/interactions/explainability", params={
                "drug_a_id": "DDInter1", "drug_b_id": "DDInter1033"})
        self.assertEqual(response.status_code, 200)
        record = response.json()["pair_specific_evidence"]["record"]
        self.assertEqual(record["severity"], "Major")
        self.assertTrue(record["pair_id"])
        self.assertIsNone(record["mechanism"])
        self.assertEqual(record["drug_a_id"], "DDInter1")
        self.assertEqual(record["drug_b_id"], "DDInter1033")

    def test_label_mention_is_not_promoted_to_a_documented_pair_interaction(self):
        label = {"source_id": "set-id", "evidence_id": "evidence-id",
                 "source_url": "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=set-id",
                 "source_version": "3", "source_date": "2025-07-07",
                 "section_name": "DRUG INTERACTIONS", "section_title": "7 DRUG INTERACTIONS",
                 "text": "This aminosalicylic acid label mentions amcinonide."}
        with patch("ddi_guard.api._text_evidence_by_drug", return_value={"DDInter75": [label]}), \
             patch("ddi_guard.api._pubmed_pair_literature", return_value={"status": "no_results", "items": []}), \
             patch("ddi_guard.api._live_dailymed_interaction_passage", return_value={"status": "no_interaction_section"}), \
             patch("ddi_guard.api._live_openfda_pair_labels", return_value={"status": "searched", "by_drug": {}}):
            body = TestClient(app).get("/api/interactions/explainability", params={
                "drug_a_id": "DDInter75", "drug_b_id": "DDInter63"}).json()
        item = body["drug_label_evidence"]["DDInter75"][0]
        self.assertEqual(item["evidence_category"], "Pair-specific label mention")
        self.assertIn("does not establish a clinical interaction", item["source_scope_note"])

    def test_openfda_pair_label_passage_is_shown_with_source_provenance(self):
        from ddi_guard.api import _live_openfda_pair_labels
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        response.read = Mock(return_value=(
            '{"results":[{"set_id":"verified-spl-id","effective_time":"20261001",'
            '"version":"2","openfda":{"generic_name":["AMINOSALICYLIC ACID"]},'
            '"drug_interactions":["Concomitant administration with amcinonide requires review of interaction evidence."]}]}').encode())
        _live_openfda_pair_labels.cache_clear()
        with patch("ddi_guard.api.urlopen", return_value=response):
            result = _live_openfda_pair_labels("DDInter75", "DDInter63")
        self.assertEqual(result["by_drug"]["DDInter75"]["status"], "pair_mention_found")
        item = result["by_drug"]["DDInter75"]["pair_mentions"][0]
        self.assertIn("amcinonide", item["text"].casefold())
        self.assertIn("verified-spl-id", item["source_url"])
        self.assertEqual(item["source_name"], "openFDA")
        _live_openfda_pair_labels.cache_clear()

        with patch("ddi_guard.api._live_openfda_pair_labels", return_value=result), \
             patch("ddi_guard.api._live_dailymed_interaction_passage", return_value={"status": "no_interaction_section"}), \
             patch("ddi_guard.api._pubmed_pair_literature", return_value={"status": "no_results", "items": []}):
            body = TestClient(app).get("/api/interactions/explainability", params={
                "drug_a_id": "DDInter75", "drug_b_id": "DDInter63"}).json()
        source_record = next(item for item in body["drug_label_evidence"]["DDInter75"]
                             if item.get("source_name") == "openFDA")
        self.assertEqual(source_record["evidence_category"], "Pair-specific label mention")
        self.assertEqual(source_record["text"], item["text"])

    def test_explainability_summary_records_actual_model_output_and_source_links(self):
        evidence = {"pair": {"drug_a": {"drug_id": "DDInter75", "name": "Aminosalicylic acid"},
                    "drug_b": {"drug_id": "DDInter63", "name": "Amcinonide"}},
                    "molecular_context": {"DDInter75": {"pubchem_cid": 4649,
                    "pubchem_url": "https://pubchem.ncbi.nlm.nih.gov/compound/4649"}},
                    "pair_specific_evidence": {"record": None}, "drug_label_evidence": {},
                    "external_search": {"pubmed": {"items": []}}, "drug_target_evidence": {}}
        prediction = {"status": "research_prediction", "drug_a": {"drug_id": "DDInter75"},
                      "drug_b": {"drug_id": "DDInter63"}, "predicted_severity": "Minor",
                      "class_probabilities": {"Minor": 0.99, "Moderate": 0.006, "Major": 0.004}}
        with patch("ddi_guard.api._build_explainability_evidence", return_value=evidence), \
             patch("ddi_guard.api._research_demo_authorized", return_value=True), \
             patch("ddi_guard.research_inference.infer_research_pair", return_value=prediction), \
             patch("ddi_guard.api.assistant_status", return_value={"configured": True}), \
             patch("ddi_guard.api.chat_completion", return_value="A. Prediction interpretation\nB. Evidence\nC. No verified pair mechanism\nD. Limits\nE. Sources") as completion:
            response = TestClient(app).post("/api/interactions/explainability/summary", json={
                "drug_a_id": "DDInter75", "drug_b_id": "DDInter63"})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "available")
        self.assertEqual(body["model_output"]["predicted_severity"], "Minor")
        self.assertTrue(any(item["source_name"] == "PubChem" for item in body["sources_used"]))
        request_context = completion.call_args.args[0][1]["content"]
        self.assertIn("DDInter75", request_context)
        self.assertIn("Minor", request_context)

    def test_explainability_ai_is_optional_and_receives_only_traceable_sources(self):
        from ddi_guard.api import AssistantRateLimitError
        evidence = {"pair": {"drug_a": {"drug_id": "DDInter75", "name": "Aminosalicylic acid"},
                    "drug_b": {"drug_id": "DDInter63", "name": "Amcinonide"}},
                    "molecular_context": {"DDInter75": {"pubchem_cid": 4649,
                    "pubchem_url": "https://pubchem.ncbi.nlm.nih.gov/compound/4649"}},
                    "pair_specific_evidence": {"record": None},
                    "drug_label_evidence": {}, "external_search": {"pubmed": {"items": []}},
                    "drug_target_evidence": {}}
        with patch("ddi_guard.api._build_explainability_evidence", return_value=evidence), \
             patch("ddi_guard.api._research_demo_authorized", return_value=True), \
             patch("ddi_guard.research_inference.infer_research_pair", return_value={"status": "research_prediction"}), \
             patch("ddi_guard.api.assistant_status", return_value={"configured": True}), \
             patch("ddi_guard.api.chat_completion", side_effect=AssistantRateLimitError("rate limited")):
            response = TestClient(app).post("/api/interactions/explainability/summary", json={
                "drug_a_id": "DDInter75", "drug_b_id": "DDInter63"})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "fallback")
        self.assertEqual(body["answer_type"], "source_grounded_fallback")
        self.assertIn("A. Prediction interpretation", body["summary"])
        self.assertIn("not established by the retrieved records", body["summary"])
        self.assertIn("not an AI-generated mechanism", body["limitations"])

    def test_explainability_panel_is_pair_scoped_below_prediction(self):
        page = TestClient(app).get("/").text
        inspect = page[page.index("async function inspectPair()"):page.index("function renderEvidencePage(")]
        self.assertIn('renderPairSummary(p, prediction), explainabilityHost.section', inspect)
        self.assertIn("loadPairExplainability(explainabilityHost", inspect)
        self.assertIn("state.selected.map((drug) => drug?.drug_id || \"\")", page)
        self.assertIn('"DDInter75", "DDInter63"', page)
        self.assertIn('makeText("h4", "Pair evidence check", "")', page)
        self.assertIn("Pair-specific source material found", page)
        self.assertIn("FDA label API:", page)
        self.assertIn("ChEMBL 34 background for", page)
        self.assertIn('summary.status === "fallback"', page)
        self.assertIn("model class probability", page)
        self.assertIn("neither establishes the pair's mechanism or clinical risk", page)

    def test_warfarin_and_aspirin_search_prioritize_exact_catalog_identity(self):
        client = TestClient(app)
        warfarin = client.get("/api/drugs/search", params={"q": "  Warfarin  "})
        self.assertEqual(warfarin.status_code, 200)
        self.assertEqual(warfarin.json()["result_count"], 0)
        self.assertEqual(warfarin.json()["results"], [])
        aspirin = client.get("/api/drugs/search", params={"q": "ASPIRIN"})
        self.assertEqual(aspirin.status_code, 200)
        self.assertEqual(aspirin.json()["result_count"], 1)
        self.assertEqual(aspirin.json()["results"][0]["drug_id"], "DDInter20")
        self.assertEqual(aspirin.json()["results"][0]["name"], "Acetylsalicylic acid")
        details = client.get("/api/drugs/DDInter20")
        self.assertEqual(details.status_code, 200)
        self.assertEqual(details.json()["molecular_formula"], "C9H8O4")
        self.assertTrue(details.json()["smiles"])
        warfarin_3d = client.get("/api/drugs/DDInter1951/structure-3d")
        self.assertEqual(warfarin_3d.status_code, 200)
        self.assertTrue(warfarin_3d.json()["available"])
        self.assertEqual(warfarin_3d.json()["conformer_type"], "rdkit_etkdg_uff_generated")
        warfarin_2d = client.get("/api/drugs/DDInter1951/structure-2d.svg")
        self.assertEqual(warfarin_2d.status_code, 200)
        self.assertEqual(warfarin_2d.headers["x-structure-dimension"], "2D")

    def test_catalog_search_empty_and_unknown_results_are_distinct(self):
        client = TestClient(app)
        blank = client.get("/api/drugs/search", params={"q": "   ", "limit": 5})
        self.assertEqual(blank.status_code, 200)
        self.assertGreater(blank.json()["result_count"], 5)
        self.assertEqual(len(blank.json()["results"]), 5)
        unknown = client.get("/api/drugs/search", params={"q": "not-a-reviewed-drug-987"})
        self.assertEqual(unknown.status_code, 200)
        self.assertEqual(unknown.json()["result_count"], 0)
        self.assertEqual(unknown.json()["results"], [])

    def test_drug_info_exposes_verified_pubchem_record_for_abaloparatide(self):
        client = TestClient(app)
        details = client.get("/api/drugs/DDInter2")
        self.assertEqual(details.status_code, 200)
        self.assertEqual(details.json()["name"], "Abaloparatide")
        self.assertEqual(details.json()["pubchem_cid"], 76943386)
        self.assertEqual(
            details.json()["structure_source_url"],
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/Abaloparatide/property/CanonicalSMILES,IsomericSMILES,MolecularFormula,MolecularWeight/JSON",
        )

    def test_usage_summary_prefers_verified_dailymed_indication(self):
        from ddi_guard.api import _verified_usage_passages

        passage = {"section_title": "INDICATIONS AND USAGE", "text": "Used for the verified example indication.",
                   "source_url": "https://dailymed.nlm.nih.gov/example", "evidence_id": "dailymed:test",
                   "source_version": "1", "label": "DailyMed · Indications and Usage"}
        with patch("ddi_guard.api._verified_usage_passages", return_value=[passage]), \
                patch("ddi_guard.api.chat_completion") as completion:
            response = TestClient(app).get("/api/drugs/DDInter2/usage-summary")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "verified")
        self.assertEqual(response.json()["summary"], passage["text"])
        self.assertEqual(response.json()["source"]["url"], passage["source_url"])
        completion.assert_not_called()

    def test_usage_summary_falls_back_to_current_dailymed_label_before_ai(self):
        passage = {"text": "FDA label indication excerpt.",
                   "source_url": "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=example",
                   "source_id": "spl-set-example", "source_version": "20261001",
                   "evidence_id": "dailymed-live:example"}
        with patch("ddi_guard.api._verified_usage_passages", return_value=[]), \
                patch("ddi_guard.api._live_dailymed_usage_passage", return_value=passage) as lookup, \
                patch("ddi_guard.api._generate_ai_usage_summary") as generate:
            response = TestClient(app).get("/api/drugs/DDInter2/usage-summary")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "source_backed")
        self.assertEqual(response.json()["summary"], passage["text"])
        self.assertIn("DailyMed", response.json()["source"]["label"])
        lookup.assert_called_once_with("Abaloparatide")
        generate.assert_not_called()

    def test_live_dailymed_lookup_extracts_current_indication_by_exact_drug_name(self):
        import json
        from unittest.mock import Mock
        from ddi_guard.api import _live_dailymed_usage_passage

        _live_dailymed_usage_passage.cache_clear()
        listing = {"data": [{"setid": "038f2461-834b-4488-9ebd-863c83eef5a7",
                             "title": "NINLARO (IXAZOMIB) CAPSULE",
                             "published_date": "Aug 17, 2026", "spl_version": 18}]}
        xml = b'''<?xml version="1.0"?><document xmlns="urn:hl7-org:v3"><section>
          <code code="34067-9"/><title>INDICATIONS AND USAGE</title>
          <text><paragraph>NINLARO is indicated with companion drugs for the test disease.</paragraph>
          <paragraph>NINLARO is indicated with companion drugs for the test disease.</paragraph></text>
        </section></document>'''
        listing_response = Mock()
        listing_response.__enter__ = Mock(return_value=listing_response)
        listing_response.__exit__ = Mock(return_value=False)
        listing_response.read.return_value = json.dumps(listing).encode()
        xml_response = Mock()
        xml_response.__enter__ = Mock(return_value=xml_response)
        xml_response.__exit__ = Mock(return_value=False)
        xml_response.read.return_value = xml
        with patch("ddi_guard.api.urlopen", side_effect=[listing_response, xml_response]):
            passage = _live_dailymed_usage_passage("Ixazomib")
        self.assertIn("test disease", passage["text"])
        self.assertEqual(passage["source_id"], "038f2461-834b-4488-9ebd-863c83eef5a7")
        self.assertIn("dailymed.nlm.nih.gov", passage["source_url"])
        self.assertTrue(passage["evidence_id"].startswith("dailymed-live:"))
        _live_dailymed_usage_passage.cache_clear()

    def test_usage_summary_ai_generation_is_identity_bound_and_cached(self):
        from ddi_guard.api import _generate_ai_usage_summary

        _generate_ai_usage_summary.cache_clear()
        answer = '{"status":"available","summary":"Abaloparatide is a parathyroid hormone-related peptide analog used to treat osteoporosis in people at high fracture risk."}'
        with patch.dict("os.environ", {"DDI_GUARD_AI_API_KEY": "test-secret", "DDI_GUARD_AI_PROVIDER": "openai"}), \
                patch("ddi_guard.api._verified_usage_passages", return_value=[]), \
                patch("ddi_guard.api._live_dailymed_usage_passage", return_value=None), \
                patch("ddi_guard.api.chat_completion", return_value=answer) as completion:
            client = TestClient(app)
            first = client.get("/api/drugs/DDInter2/usage-summary")
            second = client.get("/api/drugs/DDInter2/usage-summary")
        self.assertEqual(first.status_code, 200)
        self.assertEqual(first.json()["status"], "ai_generated")
        self.assertIn("AI-generated", first.json()["label"])
        self.assertEqual(first.json()["summary"], second.json()["summary"])
        completion.assert_called_once()
        sent = str(completion.call_args.args[0])
        self.assertIn("Abaloparatide", sent)
        self.assertIn("76943386", sent)
        self.assertNotIn("test-secret", sent)
        self.assertNotIn("3961", sent)
        _generate_ai_usage_summary.cache_clear()

    def test_usage_summary_unknown_or_ambiguous_drug_and_provider_failures_fall_back(self):
        from ddi_guard.assistant import AssistantRateLimitError, AssistantTimeoutError

        client = TestClient(app)
        self.assertEqual(client.get("/api/drugs/not-a-drug/usage-summary").status_code, 404)
        with patch("ddi_guard.api.chat_completion") as completion:
            ambiguous = client.get("/api/drugs/DDInter100/usage-summary")
        self.assertEqual(ambiguous.status_code, 200)
        self.assertEqual(ambiguous.json()["reason_code"], "identity_unverified")
        completion.assert_not_called()

        with patch.dict("os.environ", {}, clear=True), patch("ddi_guard.api.chat_completion") as completion:
            with patch("ddi_guard.api._live_dailymed_usage_passage", return_value=None):
                unconfigured = client.get("/api/drugs/DDInter2/usage-summary")
        self.assertEqual(unconfigured.json()["reason_code"], "assistant_unavailable")
        self.assertTrue(unconfigured.json()["retryable"])
        completion.assert_not_called()

        for error, expected in ((AssistantRateLimitError("private detail"), "provider_rate_limited"),
                                (AssistantTimeoutError("private detail"), "provider_timeout")):
            with patch.dict("os.environ", {"DDI_GUARD_AI_API_KEY": "test-secret", "DDI_GUARD_AI_PROVIDER": "openai"}), \
                    patch("ddi_guard.api._verified_usage_passages", return_value=[]), \
                patch("ddi_guard.api._live_dailymed_usage_passage", return_value=None), \
                    patch("ddi_guard.api.chat_completion", side_effect=error):
                response = client.get("/api/drugs/DDInter2/usage-summary")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["reason_code"], expected)
            self.assertTrue(response.json()["retryable"])
            self.assertNotIn("private detail", response.text)

        with patch.dict("os.environ", {"DDI_GUARD_AI_API_KEY": "test-secret", "DDI_GUARD_AI_PROVIDER": "openai"}), \
                patch("ddi_guard.api._verified_usage_passages", return_value=[]), \
                patch("ddi_guard.api._live_dailymed_usage_passage", return_value=None), \
                patch("ddi_guard.api.chat_completion", return_value="not-json"):
            malformed = client.get("/api/drugs/DDInter2/usage-summary")
        self.assertEqual(malformed.json()["reason_code"], "provider_error")
        self.assertTrue(malformed.json()["retryable"])

    def test_catalog_browsing_uses_offsets_and_returns_distinct_canonical_ids(self):
        client = TestClient(app)
        first = client.get("/api/drugs/search", params={"limit": 7, "offset": 0}).json()
        second = client.get("/api/drugs/search", params={"limit": 7, "offset": first["next_offset"]}).json()
        first_ids = [drug["drug_id"] for drug in first["results"]]
        second_ids = [drug["drug_id"] for drug in second["results"]]
        self.assertEqual(len(first_ids), 7)
        self.assertTrue(first["has_more"])
        self.assertEqual(first["next_offset"], 7)
        self.assertTrue(set(first_ids).isdisjoint(second_ids))
        self.assertEqual(len(set(first_ids + second_ids)), 14)
        self.assertGreater(first["result_count"], 14)

    def test_home_injects_only_public_api_base_configuration(self):
        client = TestClient(app)
        with patch.dict("os.environ", {"DDI_GUARD_PUBLIC_API_BASE_URL": "https://api.example.test"}):
            response = client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn('window.DDI_GUARD_CONFIG={"apiBaseUrl": "https://api.example.test"}', response.text)
        self.assertNotIn("DDI_GUARD_AI_API_KEY", response.text)

    def test_home_rejects_non_http_or_pathful_api_base(self):
        client = TestClient(app)
        for invalid_base in ("file:///tmp/site", "https://api.example.test/v1", "https://user:pass@example.test"):
            with self.subTest(invalid_base=invalid_base), patch.dict("os.environ", {"DDI_GUARD_PUBLIC_API_BASE_URL": invalid_base}):
                response = client.get("/")
                self.assertEqual(response.status_code, 500)
                self.assertIn("HTTP(S) origin", response.json()["detail"])

    def test_system_status_reports_catalog_local_structure_and_unchecked_viewer_separately(self):
        response = TestClient(app).get("/api/system-status")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["drug_catalog"]["available"])
        self.assertGreater(body["drug_catalog"]["drug_count"], 0)
        self.assertTrue(body["structure_service"]["available"])
        self.assertFalse(body["structure_service"]["live_external_lookup"])
        self.assertEqual(body["viewer_library"]["status"], "not_checked")

    def test_verified_and_generated_3d_conformers_and_graph_unavailable_fallback(self):
        client = TestClient(app)
        response = client.get("/api/drugs/DDInter1/structure-3d")
        self.assertEqual(response.status_code, 200)
        structure = response.json()
        self.assertTrue(structure["available"])
        self.assertEqual(structure["conformer_status"], "validated_graph_isomorphism")
        self.assertTrue(structure["canonical_structure_match"])
        self.assertEqual(len(structure["atoms"]), 21)
        self.assertGreater(len(structure["bonds"]), 0)
        self.assertTrue(structure["source_url"].startswith("https://pubchem.ncbi.nlm.nih.gov/"))
        for drug_id in ("DDInter1", "DDInter1033"):
            demo_structure = client.get(f"/api/drugs/{drug_id}/structure-3d").json()
            self.assertTrue(demo_structure["available"], drug_id)
        generated = client.get("/api/drugs/DDInter1002/structure-3d").json()
        self.assertTrue(generated["available"])
        self.assertEqual(generated["conformer_type"], "rdkit_etkdg_uff_generated")
        self.assertEqual(generated["conformer_status"], "generated_from_validated_graph")
        self.assertTrue(generated["canonical_structure_match"])
        self.assertEqual(len(generated["atoms"]), 23)
        self.assertGreater(len(generated["bonds"]), 0)
        self.assertIn("not an experimental pose", generated["provenance"])
        repeated = client.get("/api/drugs/DDInter1002/structure-3d").json()
        self.assertEqual(generated["atoms"], repeated["atoms"])
        from ddi_guard.api import _drug_catalog
        no_graph = next(drug_id for drug_id, record in _drug_catalog().items()
                        if not record["structure_available"])
        unavailable = client.get(f"/api/drugs/{no_graph}/structure-3d")
        self.assertEqual(unavailable.status_code, 200)
        self.assertFalse(unavailable.json()["available"])
        self.assertIn("No validated molecular graph", unavailable.json()["reason"])
        unavailable_svg = client.get(f"/api/drugs/{no_graph}/structure-2d.svg")
        self.assertEqual(unavailable_svg.status_code, 404)

    def test_research_results_and_system_status_use_saved_artifacts(self):
        client = TestClient(app)
        result = client.get("/api/research-results")
        self.assertEqual(result.status_code, 200)
        body = result.json()
        self.assertEqual(body["protocols"]["pair_disjoint"]["matched_test_support"], 1802)
        cold = body["protocols"]["cold_drug"]
        self.assertEqual(cold["test_class_support"]["Minor"], 14)
        self.assertEqual(cold["models"]["Full Molecular + Target + Text"]["metrics"]["macro_f1"],
                         0.36635632287806197)
        status = client.get("/api/system-status").json()
        self.assertTrue(status["backend"]["healthy"])
        self.assertFalse(status["prediction"]["ready"])
        self.assertEqual(status["prediction"]["http_status"], 503)
        self.assertTrue(status["research_prediction"]["ready"])
        self.assertEqual(status["research_prediction"]["endpoint"], "/api/research/predict")
        self.assertEqual(status["three_d_structure_coverage"]["available_conformers"], 940)

    def test_assistant_is_clear_when_unconfigured_and_accepts_only_reviewed_pair_context(self):
        client = TestClient(app)
        with patch.dict("os.environ", {}, clear=True):
            response = client.post("/api/assistant/chat", json={"messages":[{"role":"user","content":"Explain this pair"}]})
        self.assertEqual(response.status_code, 503)
        self.assertIn("Set DDI_GUARD_AI_API_KEY", response.json()["detail"])
        invalid = client.post("/api/assistant/chat", json={
            "messages":[{"role":"user","content":"Explain this pair"}],
            "selected_drug_ids":["unknown-a", "DDInter1"]})
        self.assertEqual(invalid.status_code, 404)
        same = client.post("/api/assistant/chat", json={
            "messages":[{"role":"user","content":"Explain this pair"}],
            "selected_drug_ids":["DDInter1", "DDInter1"]})
        self.assertEqual(same.status_code, 422)

    def test_assistant_mocked_provider_receives_minimal_source_context_not_prediction(self):
        client = TestClient(app)
        with patch("ddi_guard.api.chat_completion", return_value="This is an educational answer.") as completion, \
                patch.dict("os.environ", {"DDI_GUARD_AI_API_KEY":"test-secret", "DDI_GUARD_AI_PROVIDER":"openai"}):
            response = client.post("/api/assistant/chat", json={
                "messages":[{"role":"user","content":"What does the dataset record say?"}],
                "selected_drug_ids":["DDInter1951", "DDInter1164"]})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["answer_type"], "ai_generated_educational_explanation")
        self.assertFalse(body["prediction_available"])
        self.assertTrue(body["source_context"])
        sent = completion.call_args.args[0]
        prompt = " ".join(m["content"].casefold() for m in sent)
        self.assertIn("not a model prediction", prompt)
        self.assertIn("0.7473041966619568", prompt)
        self.assertIn("14", prompt)
        self.assertIn("artifacts/model_card.md", prompt)
        self.assertNotIn("test-secret", str(sent))
        self.assertEqual(completion.call_args.kwargs,
                         {"timeout_seconds": 12.0, "max_tokens": 420, "retry_rate_limit": False})
        self.assertTrue(any(source["label"] == "Paired multimodal evaluation"
                            for source in response.json()["source_context"]))
        oversized = client.post("/api/assistant/chat", json={
            "messages":[{"role":"user","content":"x"*4001}]})
        self.assertEqual(oversized.status_code, 422)

    def test_chat_rate_limit_returns_a_fast_verified_local_answer_and_cools_down(self):
        from ddi_guard.api import AssistantRateLimitError
        client = TestClient(app)
        with patch.dict("os.environ", {"DDI_GUARD_AI_API_KEY": "test-secret",
                                       "DDI_GUARD_AI_PROVIDER": "openai",
                                       "DDI_GUARD_AI_MODEL": "test-model"}), \
                patch("ddi_guard.api.chat_completion",
                      side_effect=AssistantRateLimitError("private provider detail", 45)) as completion:
            first = client.post("/api/assistant/chat", json={
                "messages": [{"role": "user", "content": "What are the evaluation metrics?"}]})
            second = client.post("/api/assistant/chat", json={
                "messages": [{"role": "user", "content": "Is production prediction enabled?"}]})
        self.assertEqual(first.status_code, 200)
        self.assertEqual(first.json()["answer_type"], "verified_local_fallback")
        self.assertEqual(first.json()["provider_status"], "rate_limited")
        self.assertIn("0.7473", first.json()["answer"])
        self.assertIn("not AI-generated", first.json()["answer"])
        self.assertTrue(first.json()["source_context"])
        self.assertEqual(second.status_code, 200)
        self.assertEqual(second.json()["provider_status"], "rate_limited_cooldown")
        self.assertIn("HTTP 503", second.json()["answer"])
        self.assertEqual(completion.call_count, 1)

    def test_frontend_exposes_all_sections_without_mock_prediction_copy(self):
        client = TestClient(app)
        page = client.get("/").text
        for page_name in ("overview", "explorer", "results", "evidence", "status", "about",
                          "riskmitigation", "report"):
            self.assertIn(f'data-page="{page_name}"', page)
        for page_name in ("molecular", "assistant", "insights", "history"):
            self.assertNotIn(f'data-page="{page_name}"', page)
            self.assertNotIn(f'id="page-{page_name}"', page)
        for removed_label in ("Research tools", "3D Molecular Lab", "Dataset & Model Insights",
                              "Saved evaluation artifacts", "Prediction History",
                              "DDI-GUARD Research Assistant"):
            self.assertNotIn(removed_label, page)
        self.assertIn('id="assistant-launcher"', page)
        self.assertIn('id="assistant-drawer"', page)
        self.assertIn('<div class="nav-label">Analysis</div>', page)
        for section in ("DDI Prediction", "Risk Assessment", "Explainability", "Risk Mitigation", "Complete Report"):
            self.assertIn(section, page)
        self.assertIn('src="/tests/capsule-for-drug-delivery-3d-rendering-KYC8XP.jpg"', page)
        self.assertIn('class="hero-image"', page)
        self.assertIn("structure-3d", page)
        self.assertIn("/api/assistant/chat", page)
        self.assertIn("HTTP 503", page)
        self.assertIn('["DDInter75", "DDInter63"]', page)
        for internal_status in (
            "Research inference available · production blocked",
            "Not API-approved",
            "The full model was trained on covered subsets, not every DDInter pair. Test performance is not used for checkpoint selection.",
        ):
            self.assertNotIn(internal_status, page)

    def test_overview_architecture_is_interactive_and_flows_top_to_bottom(self):
        page = TestClient(app).get("/").text
        self.assertIn('aria-label="DDI-GUARD architecture, flowing from drug input at the top to final report at the bottom"', page)
        self.assertEqual(page.count('class="architecture-stage"'), 10)
        for title in (
            "Drug input and identity resolution", "Verified feature sources", "Feature construction",
            "Feature encoding", "Multimodal feature fusion", "DDI severity prediction",
            "Explainability and evidence", "Risk assessment", "Risk mitigation support", "DDI-GUARD report",
            "Molecular message-passing GNN", "Relation-aware drug–target GNN", "TF-IDF text feature encoder",
            "Modality attention + gated fusion", "DDInter labels · evidence map", ">Minor</b>", ">Moderate</b>", ">Major</b>",
        ):
            self.assertIn(title, page)
        self.assertNotIn("Source-linked processing and evaluation workflow", page)
        self.assertIn('id="architecture-detail-close"', page)
        self.assertIn("highlightArchitecturePath", page)
        self.assertIn("prefers-reduced-motion: reduce", page)

    def test_frontend_contains_state_and_accessibility_guards(self):
        page = TestClient(app).get("/").text
        for guard in (
            'aria-autocomplete="list"', 'aria-expanded="false"',
            "state.searchSequence[index]", "identity !==", "state.selected.map",
            "state.chatBusy", "WEBGL_lose_context",
            "Atom colors by element", 'setAttribute("aria-pressed"',
            "async function apiFetch(", 'location.protocol === "file:"',
            "DDI_GUARD_CONFIG",
            "15000", "Drug catalog", "3D viewer library",
        ):
            self.assertIn(guard, page)
        self.assertIn('role="combobox"', page)
        self.assertIn('placeholder="Search or select Drug A"', page)
        self.assertIn('placeholder="Search or select Drug B"', page)
        self.assertIn("Load more drugs", page)
        self.assertIn("No matching drugs found", page)
        self.assertIn("aria-activedescendant", page)
        self.assertEqual(page.count("fetch("), 1, "Backend requests should use the centralized API client.")
        self.assertIn('input.value = "";', page)
        self.assertNotIn("confidence is high", page.casefold())

    def test_prediction_view_is_concise_and_detailed_evidence_stays_dedicated(self):
        page = TestClient(app).get("/").text
        self.assertNotIn('id="selected-drug-details"', page)
        self.assertNotIn("loadSelectedDrugDetails", page)
        self.assertIn('id="drug-info-cards"', page)
        pair_summary = page[
            page.index("function renderPairSummary("):
            page.index("async function inspectPair()")
        ]
        self.assertIn("renderPrediction(prediction)", pair_summary)
        self.assertNotIn("renderDrugEvidence", pair_summary)
        self.assertNotIn("dataset_annotation", pair_summary)
        self.assertIn("class_probabilities", page)
        self.assertIn('"/api/research/predict"', page)
        evidence_page = page[
            page.index("function renderEvidencePage(p)"):
            page.index('$("inspect-pair").addEventListener')
        ]
        self.assertIn("renderDrugEvidence", evidence_page)
        self.assertIn("DDInter source annotation", evidence_page)

    def test_selected_drug_cards_use_verified_metadata_and_available_images(self):
        page = TestClient(app).get("/").text
        self.assertLess(page.index('id="selected-drugs"'), page.index('id="drug-info-cards"'))
        for feature in (
            "syncDrugInfoCards()",
            "loadDrugInfoCard(index, drug",
            "molecular_formula",
            "molecular_weight",
            "indicationEvidence(details)",
            "loadDrugUsageSummary(index, drug",
            "AI-generated summary · educational, not an official indication",
            "Usage information is currently unavailable.",
            "/usage-summary",
            "structure-2d.svg",
            "structure-3d",
            "PubChem 3D conformer",
            "Retry drug details",
            "IntersectionObserver",
        ):
            self.assertIn(feature, page)
        self.assertNotIn('caption: "Drug overview"', page)
        self.assertIn("grid-template-columns: repeat(2, minmax(0, 1fr))", page)
        self.assertIn('`Interactive verified 3D molecular structure of ${drug.name}`', page)


if __name__ == "__main__":
    unittest.main()
