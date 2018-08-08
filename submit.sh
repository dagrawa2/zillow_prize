set -e

if [ "$#" != 2 ]; then
	echo "submit.sh expects two arguments-- an integer and a string"
else
	kaggle competitions submit -c zillow-prize-1 -f ./results/"$1"/submission.csv -m "$2"
fi
