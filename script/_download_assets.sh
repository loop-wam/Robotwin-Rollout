ROBOTWIN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROBOTWIN_ROOT}/assets"
python _download.py

# background_texture
unzip -o background_texture.zip
rm -rf background_texture.zip

# embodiments
unzip -o embodiments.zip
rm -rf embodiments.zip

# objects
unzip -o objects.zip
rm -rf objects.zip

cd "${ROBOTWIN_ROOT}"
echo "Configuring Path ..."
python ./script/update_embodiment_config_path.py
