# resumable queue: each stage skipped if its output exists
cd ..
st(){ name=$1; mod=$2; cls=$3; tag=$4; ep=$5; pat=$6
  [ -f runs/$name.done ] || { python3 -c "import $mod; $mod.train(epochs=$ep, patience=$pat)" > runs_$name.log 2>&1 && touch runs/$name.done; }
  [ -f res/evalsum_$tag.csv ] || python3 gen_eval.py $mod $cls runs/$name.pt $tag > gen_$name.log 2>&1; }
st cvae m_cvae CVAE CVAE 25 5
st diff m_diff Diff Diffusion 40 6
st fm m_fm FM FlowMatching 40 6
st bridge m_bridge Bridge Bridge 40 6
for extra in $(cat queue_extra.txt 2>/dev/null); do eval "$extra"; done
echo ALLDONE > queue.done
