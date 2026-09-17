#!/usr/bin/env bash
# Pull the MediaPipe runtime and the face model into the repo, so the page runs
# with no network at all. Optional — the default is the CDN (see src/tracker.js).
# Nothing here is committed: vendor/ and models/*.task are gitignored, because
# 16MB of binaries is most of what a clone would be.
#
#   npm run vendor    then open  index.html?local=1
set -euo pipefail
cd "$(dirname "$0")/.."

VER=$(grep -oE "const VERSION = '[^']+'" src/tracker.js | grep -oE "[0-9]+\.[0-9]+\.[0-9]+")
echo "vendoring @mediapipe/tasks-vision@${VER}"

mkdir -p vendor/tasks-vision models tmp-vendor
URL=$(curl -sSf "https://registry.npmjs.org/@mediapipe/tasks-vision/${VER}" \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["dist"]["tarball"])')
curl -sSf -o tmp-vendor/tv.tgz "$URL"
tar xzf tmp-vendor/tv.tgz -C tmp-vendor
rm -rf vendor/tasks-vision
mv tmp-vendor/package vendor/tasks-vision
rm -rf tmp-vendor

# The float16 model. The float32 one is twice the size for no visible gain on
# a phone; the iris points are not where the precision goes.
curl -sSf -o models/face_landmarker.task \
  'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'

du -sh vendor/tasks-vision models/face_landmarker.task
echo "done — open index.html?local=1"
