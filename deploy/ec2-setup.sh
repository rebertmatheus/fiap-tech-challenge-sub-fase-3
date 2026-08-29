#!/usr/bin/env bash
# Setup da EC2 (Amazon Linux 2023) para rodar o app via Docker Compose.
# Rode isso depois de conectar via SSH: ssh -i sua-chave.pem ec2-user@<ip-publico>
#
# Uso: cole o conteúdo direto no terminal SSH, ou copie o arquivo pra
# instância (scp) e rode: bash ec2-setup.sh
set -euo pipefail

echo "== Atualizando pacotes =="
sudo dnf update -y

echo "== Instalando Docker e git =="
sudo dnf install -y docker git
sudo systemctl enable --now docker

echo "== Instalando os plugins docker compose e buildx (AL2023 não traz por padrão) =="
sudo mkdir -p /usr/libexec/docker/cli-plugins
sudo curl -SL "https://github.com/docker/compose/releases/latest/download/docker-compose-linux-x86_64" \
  -o /usr/libexec/docker/cli-plugins/docker-compose
sudo chmod +x /usr/libexec/docker/cli-plugins/docker-compose

# "docker compose ... up --build" usa o buildx por baixo dos panos (compose v2
# exige buildx >= 0.17) — sem esse plugin o build falha com
# "compose build requires buildx 0.17.0 or later".
BUILDX_VERSION=$(curl -s https://api.github.com/repos/docker/buildx/releases/latest | grep '"tag_name"' | cut -d '"' -f4)
sudo curl -SL "https://github.com/docker/buildx/releases/download/${BUILDX_VERSION}/buildx-${BUILDX_VERSION}.linux-amd64" \
  -o /usr/libexec/docker/cli-plugins/docker-buildx
sudo chmod +x /usr/libexec/docker/cli-plugins/docker-buildx

echo "== Criando swap de 2GB (t3.micro só tem 1GB de RAM — ajuda no build) =="
if [ ! -f /swapfile ]; then
  sudo dd if=/dev/zero of=/swapfile bs=128M count=16
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
  sudo swapon /swapfile
  echo '/swapfile swap swap defaults 0 0' | sudo tee -a /etc/fstab
fi

echo "== Clonando o repositório =="
# -b develop: é onde o trabalho está de verdade. O branch default do repo
# (master) ainda não recebeu o merge, então "git clone" sem -b pega uma
# versão desatualizada, sem docker-compose.yml nem o resto do projeto.
git clone -b develop https://github.com/rebertmatheus/fiap-tech-challenge-sub-fase-3.git
cd fiap-tech-challenge-sub-fase-3

echo "== Gerando .env com uma API_KEY aleatória =="
if [ ! -f .env ]; then
  API_KEY_VALUE=$(openssl rand -hex 32)
  echo "API_KEY=${API_KEY_VALUE}" > .env
fi

echo "== Subindo os containers (build + porta 80 pro Streamlit) =="
sudo docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build

echo
echo "== Pronto =="
sudo docker compose ps
echo
echo "App deve estar acessível em http://<IP-publico-ou-Elastic-IP>/"
echo "(lembre de associar um Elastic IP à instância antes de colocar o link no entrega.txt)"
