import os
import subprocess


class FileDownloader:
    def __init__(self, url: str, dest_dir: str):
        self.url = url
        self.file_name = url.split('/')[-1]
        self.dest_dir = dest_dir

    def download(self):
        """Télécharge le fichier depuis l'URL."""
        subprocess.run(["wget", self.url], check=True)
        print(f"Fichier téléchargé : {self.file_name}")

    def extract(self):
        """Extrait le fichier tar.gz dans le répertoire cible."""
        subprocess.run(["tar", "xf", self.file_name, "-C", self.dest_dir], check=True)
        print(f"Fichier extrait dans : {self.dest_dir}")


def main():
    url = "https://s3.amazonaws.com/opennmt-trainingdata/toy-ende.tar.gz"
    dest_dir = "./data"

    # Créer une instance de FileDownloader
    downloader = FileDownloader(url, dest_dir)

    # Exécuter les étapes de téléchargement, extraction et changement de répertoire
    downloader.download()
    downloader.extract()

if __name__ == "__main__":
    main()
