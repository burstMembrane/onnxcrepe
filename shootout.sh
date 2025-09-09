
CMD_ONE="python -m torchcrepe --audio_files testdata/test_60s.wav --output_files f0 --gpu 0"
CMD_TWO="curl -X POST \
    --unix-socket /tmp/crepetrt.sock \
    http://localhost/predict \
    -F 'audio_file=@testdata/test_60s.wav' \
    -F 'return_periodicity=true' \
    -F 'format=json'"

BRANCH=`git branch --show-current`

if ! [ -d "benchmarks" ]; then
    mkdir benchmarks
fi

hyperfine --warmup 3 "$CMD_ONE" "$CMD_TWO"  --export-markdown benchmarks/shootout_${BRANCH}.md
