# Documentação da API (backend)

- `openapi.json` — schema OpenAPI/Swagger exportado do backend (`GET /openapi.json`). Também dá pra ver interativamente em `http://localhost:8000/docs` (Swagger UI) com o backend rodando.
- `postman_collection.json` — coleção do Postman com os 3 endpoints (`/health`, `/defaults`, `/predict`) e 2 cenários de exemplo pro `/predict` (perfil mediano e perfil de alto risco), mais um caso de validação (aprovado > inscrito, espera 422).
- `postman_environment.json` — environment do Postman com `base_url` (`http://localhost:8000`) e `api_key` (**vazio de propósito** — nunca commitamos a chave real no git).

## Como usar

1. No Postman: **Import** → selecione `postman_collection.json` e `postman_environment.json`.
2. Selecione o environment "Evasão de Estudantes - Local" no canto superior direito.
3. Edite a variável `api_key` do environment e cole o valor de `API_KEY` do seu `.env` local (na raiz do projeto).
4. Com o backend rodando (`podman-compose up -d` ou `docker compose up -d`, com a porta 8000 publicada), rode as requisições.

Sempre que `backend/main.py` mudar, regenere o `openapi.json`:
```
curl -s http://localhost:8000/openapi.json | python3 -m json.tool > docs/api/openapi.json
```
