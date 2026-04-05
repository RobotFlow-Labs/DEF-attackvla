# Rule: NEVER Block the Chat with Sleep

## BANNED
```bash
# NEVER do any of these — they block the chat for minutes:
sleep 300 && tail -5 train.log
sleep 400 && nvidia-smi
sleep 120 && check something
Bash(sleep 240 && tail -30 logfile)
```

## WHY
When you run `sleep` inside a Bash call, the entire chat is blocked. You cannot:
- Receive messages from the user or supervisor
- React to crashes or OOMs
- Be interrupted or redirected
- Process urgent instructions

You become DEAF for the entire sleep duration.

## INSTEAD

### Check training status (instant, <2 seconds):
```bash
bash /mnt/forge-data/scripts/gpu-visor.sh
bash /mnt/forge-data/scripts/gpu-visor.sh --module YOUR_MODULE
```

### Launch training (survives disconnects):
```bash
nohup your_training_command > logfile 2>&1 & disown
```

### If you MUST wait, use background mode:
```bash
# Use run_in_background=true on the Bash tool
# This runs the command without blocking the chat
Bash(command, run_in_background=true)
```

### Monitor loop (non-blocking):
```bash
# Check every 2 minutes, 5 times
for i in 1 2 3 4 5; do
  bash /mnt/forge-data/scripts/gpu-visor.sh --module YOUR_MODULE
  sleep 120  # OK inside a background Bash only
done
```
Run the above with `run_in_background=true`.

## SUMMARY
- `sleep` in foreground Bash = BANNED
- `gpu-visor.sh` for instant status = MANDATORY
- `nohup + disown` for training = MANDATORY
- `run_in_background=true` for long waits = ALLOWED
