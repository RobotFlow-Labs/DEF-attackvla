base=/cpfs01/shared/optimal/vla_ptm/LIBERO/experiments

cd $base
for log in $(find . -name "*.txt"); do
    echo ------------- 🔥 $log
    tail -n 4 $log
done