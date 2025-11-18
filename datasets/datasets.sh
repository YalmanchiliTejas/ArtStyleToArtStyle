#!/bin/bash

set -e

DATA_DIR="."   # current directory
BASE_URL="http://efrosgans.eecs.berkeley.edu/cyclegan/datasets"

DATASETS=(
  "monet2photo"
  "cezanne2photo"
  "ukiyoe2photo"
  "vangogh2photo"
)

for NAME in "${DATASETS[@]}"; do
  ZIP_FILE="${DATA_DIR}/${NAME}.zip"
  TARGET_DIR="${DATA_DIR}/${NAME}"

  if [ -d "${TARGET_DIR}" ]; then
    echo "[skip] ${TARGET_DIR} already exists"
    continue
  fi

  URL="${BASE_URL}/${NAME}.zip"
  echo "[download] ${URL}"
  curl -L "${URL}" -o "${ZIP_FILE}"

  echo "[unzip] ${ZIP_FILE} -> ${DATA_DIR}"
  unzip -q "${ZIP_FILE}" -d "${DATA_DIR}"

  rm "${ZIP_FILE}"
  echo "[done] ${NAME} -> ${TARGET_DIR}"
done

echo "[all done] datasets are under ${DATA_DIR}/"
