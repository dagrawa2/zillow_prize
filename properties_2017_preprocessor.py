import pandas as pd
import numpy as np
from uszipcode import ZipcodeSearchEngine

print("Loading data")
property_data = pd.read_csv('data/properties_2017.csv')

# ### Pools & Hot tubs
print("Preprocessing pools and hot tubs features")

# "0 pools"
property_data.poolcnt.fillna(0,inplace = True)
# "0 hot tubs or spas"
property_data.hashottuborspa.fillna(0,inplace = True)
# Convert "True" to 1
property_data.hashottuborspa.replace(to_replace = True, value = 1,inplace = True)
# Set properties that have a pool but no info on poolsize equal to the median poolsize value.
property_data.loc[property_data.poolcnt==1, 'poolsizesum'] = property_data.loc[property_data.poolcnt==1, 'poolsizesum'].fillna(property_data[property_data.poolcnt==1].poolsizesum.median())
# "0 pools" = "0 sq ft of pools"
property_data.loc[property_data.poolcnt==0, 'poolsizesum']=0
# "0 pools with a spa/hot tub"
property_data.pooltypeid2.fillna(0,inplace = True)
# "0 pools without a hot tub"
property_data.pooltypeid7.fillna(0,inplace = True)
# Drop redundant feature
property_data.drop('pooltypeid10', axis=1, inplace=True)


# ### Fireplace Data
print("Preprocessing fireplace features")

# If "fireplaceflag" is "True" and "fireplacecnt" is "NaN", we will set "fireplacecnt" equal to the median value of "1".
property_data.loc[(property_data['fireplaceflag'] == True) & (property_data['fireplacecnt'].isnull()), ['fireplacecnt']] = 1
# If 'fireplacecnt' is "NaN", replace with "0"
property_data.fireplacecnt.fillna(0,inplace = True)
# If "fireplacecnt" is 1 or larger "fireplaceflag" is "NaN", we will set "fireplaceflag" to "True".
property_data.loc[(property_data['fireplacecnt'] >= 1.0) & (property_data['fireplaceflag'].isnull()), ['fireplaceflag']] = True
property_data.fireplaceflag.fillna(0,inplace = True)
# Convert "True" to 1
property_data.fireplaceflag.replace(to_replace = True, value = 1,inplace = True)


# ### Garage Data
print("Preprocessing garage features")

property_data.garagecarcnt.fillna(0,inplace = True)
property_data.garagetotalsqft.fillna(0,inplace = True)


# ### Tax Data Delinquency
print("Preprocessing tax features")

# There are two features related to tax delinquency:
# * "`taxdelinquencyflag`" - Property taxes for this parcel are past due as of 2015.
# * "`taxdelinquencyyear`" - Year for which the unpaid property taxes were due.

# Replace "NaN" with "0"
property_data.taxdelinquencyflag.fillna(0,inplace = True)

# Change "Y" to "1"
property_data.taxdelinquencyflag.replace(to_replace = 'Y', value = 1,inplace = True)

# Drop "taxdelinquencyyear"
property_data.drop('taxdelinquencyyear', axis=1, inplace=True)


# ### The Rest
print("Preprocessing the rest of the features")

# * "`storytypeid`" - Numerical ID that describes all types of homes. Mostly missing, so we should drop this category. Crazy idea would be to try and integrate street view of each home, and use image recognition to classify each type of story ID.          
# Drop "storytypeid"
property_data.drop('storytypeid', axis=1, inplace=True)

# * "`basementsqft`" - Square footage of basement. Mostly missing, suggesting no basement, so we will replace "NaN" with "0".
# Replace "NaN" with 0, signifying no basement.
property_data.basementsqft.fillna(0,inplace = True)

# * "`yardbuildingsqft26 `" - Storage shed square footage. We can set "NaN" values to "0". Might be useful to change this to a categorical category of just "1"s and "0"s (has a shed vs doesn't have a storage shed), but some of the sheds are enormous and others are tiny, so we will keep the actual square footage.
# Replace 'yardbuildingsqft26' "NaN"s with "0".
property_data.yardbuildingsqft26.fillna(0,inplace = True)

# * "`architecturalstyletypeid`" - What is the architectural style of the house? Examples: ranch, bungalow, Cape Cod, etc. Because this is only present in a small fraction of the homes, I'm going to drop this category. (Idea: One can also assume that most homes in the same neighborhood have the same style. Could also try image recognition.)
# Drop "architecturalstyletypeid"
property_data.drop('architecturalstyletypeid', axis=1, inplace=True)

# * "`typeconstructiontypeid`" - What material is the house made out of? Missing in a bunch, so probably drop category. Would be very difficult image recognition problem.
# * "`finishedsquarefeet13`" - Perimeter of living area. This seems more like describing the shape of the house and is closely related to the square footage. I recommend dropping the category.
# Drop "typeconstructiontypeid" and "finishedsquarefeet13"
property_data.drop('typeconstructiontypeid', axis=1, inplace=True)
property_data.drop('finishedsquarefeet13', axis=1, inplace=True)

# * "`buildingclasstypeid`" - Describes the internal structure of the home. Not a lot of information gained and present in less than 1% of properties. I will drop.# Drop "buildingclasstypeid"
property_data.drop('buildingclasstypeid', axis=1, inplace=True)

# * "`decktypeid`" - Type of deck (if any) on property. Looks like a value is either "66.0" or "NaN". I will keep this feature and change the "66.0" to "1" for "Yes" and "NaN" to "0" for "No".
# Change "decktypeid" "Nan"s to "0"
property_data.decktypeid.fillna(0,inplace = True)
# Convert "decktypeid" "66.0" to "1"
property_data.decktypeid.replace(to_replace = 66.0, value = 1,inplace = True)

# * "`finishedsquarefeet6`" - Base unfinished and finished area. Not sure what this means. Seems like it gives valuable information, but replacing "NaN"s with "0"s would be incorrect. Perhaps it is a subset of other categories. Probably drop, but TBD.
# * "`finishedsquarefeet15`" - Total area. Should be equal to sum of all other finishedsquarefeet categories.
# * "`finishedfloor1squarefeet`" - Sq footage of first floor. Could cross check this with number of stories.
# * "`finishedsquarefeet50`" - Identical to above category? Drop one of them. Duplicate.
# * "`finishedsquarefeet12`" - Finished living area.
# * "`calculatedfinishedsquarefeet`" - Total finished living area of home.

# "`finishedsquarefeet6`" is rarely present, and even when it is present, it is equal to "`calculatedfinishedsquarefeet`". Because of this, we will drop it. Same scenario with "`finishedsquarefeet12`", so we will drop that as well. "`finishedsquarefeet50`" is identical to "`finishedfloor1squarefeet`", so we will also drop "`finishedfloor1squarefeet`".
# Drop "finishedsquarefeet6"
property_data.drop('finishedsquarefeet6', axis=1, inplace=True)

# Drop "finishedsquarefeet12"
property_data.drop('finishedsquarefeet12', axis=1, inplace=True)

# Drop "finishedfloor1squarefeet"
property_data.drop('finishedfloor1squarefeet', axis=1, inplace=True)

# Of the six "squarefeet" categories listed above, we dropped three and are left with these three:
# * "`calculatedfinishedsquarefeet`" - Present in 98%. Total finished living area of home. Let's fill the rest with the median values.
# * "`finishedsquarefeet15`" - Present in 6.4%. Most cases, it is equal to "`calculatedfinishedsquarefeet`", so we will fill in the "NaN" values with the value of "`calculatedfinishedsquarefeet`". Total area. Should be equal to sum of all other finishedsquarefeet categories.
# * "`finishedsquarefeet50`" - If "`numberofstories`" is equal to "1", then we can replace the "NaN"s with the "`calculatedfinishedsquarefeet`" value. Fill in the rest with the average values.

# Replace "NaN" "calculatedfinishedsquarefeet" values with mean.
property_data['calculatedfinishedsquarefeet'].fillna((property_data['calculatedfinishedsquarefeet'].mean()), inplace=True)

# drop "finishedsquarefeet15" values since present in only 6.4 % and when present, is mostly equal to calculatedfinishedsquarefeet.
property_data.drop("finishedsquarefeet15", axis=1, inplace=True)

# missing numberofstories seems to imply 1 story
property_data.numberofstories.fillna(1,inplace = True)

# If "numberofstories" is equal to "1", then we can replace the "NaN"s with the "calculatedfinishedsquarefeet" value. Fill in the rest with the average values.
property_data.loc[property_data['numberofstories'] == 1.0,'finishedsquarefeet50'] = property_data['calculatedfinishedsquarefeet']
property_data['finishedsquarefeet50'].fillna((property_data['finishedsquarefeet50'].mean()), inplace=True)

# * "`yardbuildingsqft17`" - Patio in yard. Do same as storage shed category.
# Replace 'yardbuildingsqft17' "NaN"s with "0".
property_data.yardbuildingsqft17.fillna(0,inplace = True)

# Bathrooms

bathroomcnt = property_data['bathroomcnt'].value_counts().idxmax()
property_data['bathroomcnt'] = property_data['bathroomcnt'].fillna(bathroomcnt)

property_data.threequarterbathnbr.fillna(0, inplace=True)

property_data.drop("fullbathcnt", axis=1, inplace=True)
property_data.drop("calculatedbathnbr", axis=1, inplace=True)

# * "`airconditioningtypeid`" - If "NaN", change to "5" for "None".
property_data.airconditioningtypeid.fillna(5,inplace = True)

# * "`regionidneighborhood`" - Neighborhood. Could fill in blanks. Would need a key that maps lat & longitude regions with specific neighborhoods. Because "`longitude`" and "`latitude`" essentially provide this information, we will drop "`regionidneighborhood`".
# Drop "regionidneighborhood"
property_data.drop('regionidneighborhood', axis=1, inplace=True)

# * "`heatingorsystemtypeid`" - Change "NaN" to "13" for "None"
property_data.heatingorsystemtypeid.fillna(13,inplace = True)

# * "`buildingqualitytypeid`" - Change "NaN" to most common value.
# Fill in "NaN" "buildingqualitytypeid" with most common
buildingqual = property_data['buildingqualitytypeid'].value_counts().idxmax()
property_data['buildingqualitytypeid'] = property_data['buildingqualitytypeid'].fillna(buildingqual)

# * "`unitcnt`" - Number of units in a property. Change "NaN" to "1"
property_data.unitcnt.fillna(1,inplace = True)

# * "`propertyzoningdesc`" - This seems like a very error-ridden column with so many unique values. It may provide some valuable info, so lets just fill the "NaN" with the most common value.
# Fill in "NaN" "propertyzoningdesc" with most common
propertyzoningdesc = property_data['propertyzoningdesc'].value_counts().idxmax()
property_data['propertyzoningdesc'] = property_data['propertyzoningdesc'].fillna(propertyzoningdesc)

# * "`lotsizesquarefeet`" - Area of lot in square feet. Fill "NaN" with average.
property_data['lotsizesquarefeet'].fillna((property_data['lotsizesquarefeet'].mean()), inplace=True)

# * "`censustractandblock`" & "`rawcensustractandblock`" - Census tract and block ID combined. Look like duplicate values. I think we should drop these because they are related to location which is covered by "`longitude`" and "`latitude`". Let's view the values first.
# These look like information we might be able to work with. Let's just drop "`censustractandblock`" because it is the same as "`rawcensustractandblock`".

# Drop "censustractandblock"
property_data.drop('censustractandblock', axis=1, inplace=True)

# * "`landtaxvaluedollarcnt`" - Assessed value of land area of parcel.
# * "`structuretaxvaluedollarcnt`" - Assessed value of built structure on land.
# * "`taxvaluedollarcnt`" - Total tax assessed value of property. "structuretax..." + "landtax...".
# * "`taxamount`" - Total property tax assessed for assessment year.

# * "`landtaxvaluedollarcnt`" - We can fill in the "NaN"s with "0". It appears some properties do not have own any land. An example of this could be an apartment in large building where only the structurevalue would exist.
# * "`structuretaxvaluedollarcnt`" - Same as "`landtaxvaluedollarcnt`", but opposite. An example of a "NaN" in this category would be an empty lot.
# * "`taxvaluedollarcnt`" - We can just fill in the "NaN" values with the average.
# * "`taxamount`" - We should calculate a new category called 'taxpercentage' where we divide the taxamount by the 'taxvaluedollarcnt', then we can fill in the "NaN" values with the average tax percentage.

property_data.landtaxvaluedollarcnt.fillna(0,inplace = True)
property_data.structuretaxvaluedollarcnt.fillna(0,inplace = True)
property_data['taxvaluedollarcnt'].fillna((property_data['taxvaluedollarcnt'].mean()), inplace=True)

property_data['taxpercentage'] = property_data['taxamount'] / property_data['taxvaluedollarcnt']
property_data['taxpercentage'].fillna((property_data['taxpercentage'].mean()), inplace=True)

# Now we will drop "`taxamount`" because we have replaced it with "`taxpercentage`".
# Drop "taxamount"
property_data.drop('taxamount', axis=1, inplace=True)


# * "`yearbuilt`" - Year home was built. We can just fill in the "NaN" values with the most common value.
# Fill in "NaN" "yearbuilt" with most common
yearbuilt = property_data['yearbuilt'].value_counts().idxmax()
property_data['yearbuilt'] = property_data['yearbuilt'].fillna(yearbuilt)

# Fill in "fips" "NaN"s
fips = property_data['fips'].value_counts().idxmax()
property_data['fips'] = property_data['fips'].fillna(fips)

# Fill in "propertylandusetypeid" "NaN"s
propertylandusetypeid = property_data['propertylandusetypeid'].value_counts().idxmax()
property_data['propertylandusetypeid'] = property_data['propertylandusetypeid'].fillna(propertylandusetypeid)

# Fill in "rawcensustractandblock" "NaN"s
#rawcensustractandblock = property_data['rawcensustractandblock'].value_counts().idxmax()
#property_data['rawcensustractandblock'] = property_data['rawcensustractandblock'].fillna(rawcensustractandblock)

# Fill in "assessmentyear" "NaN"s
assessmentyear = property_data['assessmentyear'].value_counts().idxmax()
property_data['assessmentyear'] = property_data['assessmentyear'].fillna(assessmentyear)

# Fill in "bedroomcnt" "NaN"s
bedroomcnt = property_data['bedroomcnt'].value_counts().idxmax()
property_data['bedroomcnt'] = property_data['bedroomcnt'].fillna(bedroomcnt)

# Fill in "roomcnt" "NaN"s
roomcnt = property_data['roomcnt'].value_counts().idxmax()
property_data['roomcnt'] = property_data['roomcnt'].fillna(roomcnt)

# Fill in "propertycountylandusecode" "NaN"s
propertycountylandusecode = property_data['propertycountylandusecode'].value_counts().idxmax()
property_data['propertycountylandusecode'] = property_data['propertycountylandusecode'].fillna(propertycountylandusecode)

##

latitude = property_data.latitude.value_counts().idxmax()
property_data.latitude.fillna(latitude, inplace=True)
longitude = property_data.longitude.value_counts().idxmax()
property_data.longitude.fillna(longitude, inplace=True)

print("Imputing zipcodes")
inds = np.arange(property_data.shape[0])
inds = inds[property_data.regionidzip.isnull().values]
print(len(inds), " zipcodes to impute")
search = ZipcodeSearchEngine()
zipcodes = np.array([int(search.by_coordinate(property_data.latitude.iloc[i]*1e-6, property_data.longitude.iloc[i]*1e-6, radius=50, returns=1)[0]["Zipcode"]) for i in inds])
property_data.regionidzip.loc[inds] = zipcodes

print("Imputing county")
mask = property_data.regionidcounty.isnull().values
zips = np.unique(property_data.regionidzip.loc[mask].values)
for z in zips:
	mask_z = (property_data.regionidzip==z).values
	counties, counts = np.unique(property_data.regionidcounty.loc[np.logical_and(np.logical_not(mask), mask_z)].values, return_counts=True)
	county = counties[counts.argmax()]
	property_data.regionidcounty.loc[np.logical_and(mask, mask_z)] = county

print("Imputing city")
mask = property_data.regionidcity.isnull().values
zips = np.unique(property_data.regionidzip.loc[mask].values)
zips_no_city = []
for z in zips:
	mask_z = (property_data.regionidzip==z).values
	cities, counts = np.unique(property_data.regionidcity.loc[np.logical_and(np.logical_not(mask), mask_z)].values, return_counts=True)
	if len(counts) > 0:
		city = cities[counts.argmax()]
		property_data.regionidcity.loc[np.logical_and(mask, mask_z)] = city
else:
	zips_no_city.append(z)

print(len(zips_no_city), " zipcodes with no city")
print("Imputing missing city with mean")
city = int(property_data.regionidcity.mean())
print("Confirming that imputed city does not already exist")
if (property_data.regionidcity==city).sum() == 0:
	print("Confirmed")
property_data.regionidcity.fillna(city, inplace=True)


"""
# ### Create New Features
print("Creating new features")

property_data['taxpersqft'] = property_data['taxvaluedollarcnt'] / property_data['calculatedfinishedsquarefeet']
property_data['bathpersqft'] = property_data['bathroomcnt'] / property_data['calculatedfinishedsquarefeet']
property_data['roompersqft'] = property_data['roomcnt'] / property_data['calculatedfinishedsquarefeet']
property_data['bedroompersqft'] = property_data['bedroomcnt'] / property_data['calculatedfinishedsquarefeet']
"""

print("Saving preprocessed data")
property_data.to_csv('preprocessed/properties_2017.csv', index=False)

print("Done")
