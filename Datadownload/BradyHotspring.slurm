The Bradys Hotspring data can be accessed through: https://data.openei.org/s3_viewer?bucket=nrel-pds-porotomo&limit=100&prefix=DAS%2FH5%2FDASH%2F20160314%2F&offset=1000
The following shows a sample program for downloading this data throught the command line. This needs to have "aws cli" installed.

"""
#!/bin/bash
#SBATCH --job-name="Download BradyHotspring data"
#SBATCH --time=48:00:00
#SBATCH --mem=12GB
#SBATCH --export=ALL


~/bins/bin/aws s3 cp --no-sign-request s3://nrel-pds-porotomo/DAS/H5/DASH/20160313/ ~/BradyHotspring/03_13_2016/ --recursive
~/bins/bin/aws s3 cp --no-sign-request s3://nrel-pds-porotomo/DAS/H5/DASH/20160314/ ~/BradyHotspring/03_14_2016/ --recursive

"""

This is the template for the command above:

"aws cli" s3 cp --no-sign-request s3:/"directory to file(s)(similar to arrangement in link above)" "location to save data to" --recursive