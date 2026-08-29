# Deploy na EC2

## Pré-requisitos (feito no console AWS)

- Instância EC2: Amazon Linux 2023, `t3.micro`, 20GB gp3.
- Security Group: porta 22 (SSH) e 80 (HTTP) liberadas. Porta 8000 (backend) **não** liberada — o backend não deve ficar acessível publicamente.
- Key pair associado, Elastic IP alocado e associado à instância (senão o IP muda se ela reiniciar).

## Passos

1. Conecte via SSH:
   ```
   ssh -i sua-chave.pem ec2-user@<ip-publico-ou-elastic-ip>
   ```
2. Copie o conteúdo de [`ec2-setup.sh`](ec2-setup.sh) e cole no terminal SSH (ou `scp` o arquivo e rode `bash ec2-setup.sh`).
3. O script instala Docker, cria um swap de 2GB, clona o repositório, gera um `.env` com uma `API_KEY` aleatória e sobe os containers com `docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build` (essa combinação de arquivos publica o Streamlit na porta 80; ver `docker-compose.prod.yml` na raiz).
4. Acesse `http://<ip-publico-ou-elastic-ip>/` — deve abrir o app.

## Redeploy (depois de um `git push` com mudanças)

```
cd fiap-tech-challenge-sub-fase-3
git pull
sudo docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
```

## Não desligar

A instância precisa ficar de pé até a correção. Não parar/desligar depois de gravar o vídeo.
