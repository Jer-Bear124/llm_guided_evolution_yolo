# For cleaning up the output files from the Guided Evolution
rm -rf slurm-*.out
rm -rf 0
rm -rf sota/ultralytics/ultralytics/cfg/models/llm/models/*
rm -rf sota/ultralytics/results/*


# Run this if you need to clean up everythin, otherwise, comment it out
rm -rf first_test/*
rm -rf templates/FixedPrompts/roleplay/mutant*.txt
rm -rf templates/FixedPrompts/concise/mutant*.txt
rm -rf runs/detect/*