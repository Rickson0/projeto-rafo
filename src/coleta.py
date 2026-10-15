import requests, time, json, os

BASE = "https://wob.coppermind.net/api"
HEADERS = {
    "User-Agent": "RAFO-Sentiment/0.1 (+https://github.com/Rickson/rafo-project; ricksonkeu@gmail.com)",
    "Accept": "application/json",
}
CACHE = "wobs.json"

def fetch_all(endpoint, page_size=50):
    url = f"{BASE}/{endpoint}/?page_size={page_size}"
    todas = []
    pagina = 1

    while url:
        for tentativa in range(3):
            try:
                r = requests.get(url, headers=HEADERS, timeout=15)
                if r.status_code == 429:
                    print(f"429 na página {pagina}, esperando 30s...")
                    time.sleep(30)
                    continue
                r.raise_for_status()
                break
            except requests.RequestException as e:
                print(f"Erro na página {pagina}: {e}. Tentando de novo...")
                time.sleep(5)
        else:
            raise RuntimeError(f"Falhou de vez na página {pagina}")

        data = r.json()
        todas.extend(data["results"])
        print(f"Página {pagina}: +{len(data['results'])} (total {len(todas)}/{data['count']})")

        url = data["next"]
        pagina += 1
        time.sleep(0.5)

    return todas

if os.path.exists(CACHE):
    print("Carregando do cache...")
    entries = json.load(open(CACHE))
else:
    entries = fetch_all("entry")
    json.dump(entries, open(CACHE, "w"), ensure_ascii=False)
    print(f"Salvo em {CACHE}")

print(f"Total carregado: {len(entries)}")