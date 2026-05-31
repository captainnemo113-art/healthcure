from app import app


def test_pages_load():
    app.config["TESTING"] = True
    client = app.test_client()

    for path in ["/", "/heart", "/diabetes", "/pneumonia", "/healthz"]:
        response = client.get(path)
        assert response.status_code == 200

