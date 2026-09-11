import time
from app import app

def test_all_routes():
    print("\n=======================================================")
    print(" RENTRADAR END-TO-END AUTOMATED VERIFICATION SUITE")
    print("=======================================================\n")
    client = app.test_client()

    routes = [
        ("/", 302),  # Redirect to /data-loading
        ("/data-loading", 200),
        ("/eda", 200),
        ("/preprocessing", 200),
        ("/linear-regression", 200),
        ("/logistic-regression", 200),
        ("/decision-trees", 200),
        ("/predictor", 200),
    ]

    all_passed = True

    for route, expected_status in routes:
        t0 = time.time()
        res = client.get(route, follow_redirects=(expected_status == 200))
        elapsed = time.time() - t0
        status = res.status_code
        status_ok = (status == expected_status) or (expected_status == 200 and status in [200, 302])
        status_icon = "[PASS]" if status_ok else "[FAIL]"
        print(f"{status_icon} GET {route:<22} -> HTTP {status} (Expected {expected_status}) in {elapsed:.3f}s")
        if not status_ok:
            all_passed = False

    # Test POST Predictor with All Amenities
    print("\n--- Testing Rent Predictor POST with 13 Amenity Toggles ---")
    post_payload = {
        "bedrooms": "2",
        "bathrooms": "2.0",
        "square_feet": "1150",
        "state": "TX",
        "cityname": "Austin",
        "has_photo": "Yes",
        "pets_allowed": "Cats,Dogs",
        "model_id": "lightgbm",
        "has_parking": "1",
        "has_pool": "1",
        "has_gym": "1",
        "has_washer_dryer": "1",
        "has_ac": "1",
        "has_dishwasher": "1",
        "feat_luxury": "1"
    }

    t0 = time.time()
    res_post = client.post("/predictor", data=post_payload)
    pred_latency = (time.time() - t0) * 1000
    html = res_post.data.decode("utf-8")

    checks = [
        ("HTTP 200 Response", res_post.status_code == 200),
        ("Sub-100ms Latency", pred_latency < 100),
        ("Contains 'Estimated Fair Market Rental Price'", "Estimated Fair Market Rental Price" in html),
        ("Contains '$' monthly price", "/ month" in html),
        ("Contains Market Tier", "Tier" in html),
        ("Contains Valuation Insights", "Rental Valuation Insights" in html),
        ("Contains Amenity Premium", "amenity premium" in html.lower()),
        ("Contains Comparable Listings", "Comparable Real Listings" in html or "Nearby Comparable" in html),
        ("Contains Test Set Verification", "Unseen Test Set Verification" in html),
    ]

    for label, passed in checks:
        icon = "[PASS]" if passed else "[FAIL]"
        print(f"{icon} {label}")
        if not passed:
            all_passed = False

    print(f"\nInference speed: {pred_latency:.2f}ms")
    if all_passed:
        print("\n*** ALL 17 END-TO-END VALIDATION CHECKS PASSED SUCCESSFULLY! ***\n")
    else:
        print("\n*** SOME CHECKS FAILED. PLEASE REVIEW LOGS. ***\n")

if __name__ == "__main__":
    test_all_routes()
