import os
import sys
import shutil
import tempfile
import urllib.request
import zipfile

def get_site_packages():
    """Récupère automatiquement le chemin de site-packages pour l'environnement actif."""
    try:
        import site
        site_packages = site.getsitepackages()
        return site_packages[0] if site_packages else None
    except AttributeError:
        return None

def download_and_extract():
    """Télécharge et extrait le dossier 'data' depuis GitHub."""
    url = "https://github.com/ClaudeCoulombe/FrenchLefffLemmatizer/archive/refs/heads/master.zip"
    temp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(temp_dir, "french_lefff_lemmatizer.zip")
    
    print("🔽 Téléchargement du fichier ZIP...")
    urllib.request.urlretrieve(url, zip_path)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(temp_dir)
    
    extracted_path = os.path.join(temp_dir, "FrenchLefffLemmatizer-master", "french_lefff_lemmatizer", "data")
    
    if not os.path.exists(extracted_path):
        print("❌ Erreur : Impossible de trouver le dossier 'data' après extraction.")
        sys.exit(1)
    
    return extracted_path, temp_dir

def main():
    print("🔍 Détection du site-packages...")
    site_packages = get_site_packages()
    
    if not site_packages:
        print("❌ Impossible de détecter le dossier site-packages.")
        sys.exit(1)

    target_path = os.path.join(site_packages, "french_lefff_lemmatizer", "data")

    print(f"🎯 Chemin de destination : {target_path}")

    if os.path.exists(target_path):
        print("✅ Le dossier 'data' existe déjà. Opération annulée.")
        sys.exit(0)

    extracted_data_path, temp_dir = download_and_extract()

    print("🚀 Copie du dossier 'data' vers site-packages...")
    shutil.copytree(extracted_data_path, target_path)

    print("🎉 Installation terminée avec succès !")

    # Nettoyage du dossier temporaire
    shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()
