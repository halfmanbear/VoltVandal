python voltvandal.py run --gpu 0 --out .\artifacts `
  --mode vlock --target-voltage-mv 950 `
  --step-mhz 15 --max-steps 25 `
  --doloming .\doloMing\stress.py `
  --doloming-modes matrix,frequency-max `
  --multi-stress-seconds 30 `
  --temp-limit-c 85 --hotspot-limit-c 95 `
  --start-freq-mhz 2000 `
  --power-limit-pct 115 `