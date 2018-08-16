set -e

if [ "$#" != 2 ]; then
	echo "submit.sh expects two string arguments-- a path and a comment"
else
	kaggle competitions submit -c zillow-prize-1 -f "$1" -m "$2"
fi
