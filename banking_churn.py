{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "284b1296",
   "metadata": {},
   "outputs": [],
   "source": [
    "%matplotlib inline\n",
    "import pandas as pd\n",
    "#from  matplotlib import plotly as plt\n",
    "import numpy as np\n",
    "import warnings\n",
    "warnings.filterwarnings(\"ignore\")\n",
    "\n",
    "url = \"https://raw.githubusercontent.com/Livuza/ADS-April-2021/main/Assignments/Assignment%202/banking_churn.csv\"\n",
    "banking_churn = pd.read_csv(url)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "f365b273",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>RowNumber</th>\n",
       "      <th>CustomerId</th>\n",
       "      <th>Surname</th>\n",
       "      <th>CreditScore</th>\n",
       "      <th>Geography</th>\n",
       "      <th>Gender</th>\n",
       "      <th>Age</th>\n",
       "      <th>Tenure</th>\n",
       "      <th>Balance</th>\n",
       "      <th>NumOfProducts</th>\n",
       "      <th>HasCrCard</th>\n",
       "      <th>IsActiveMember</th>\n",
       "      <th>EstimatedSalary</th>\n",
       "      <th>Exited</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>15634602</td>\n",
       "      <td>Hargrave</td>\n",
       "      <td>619</td>\n",
       "      <td>France</td>\n",
       "      <td>Female</td>\n",
       "      <td>42</td>\n",
       "      <td>2</td>\n",
       "      <td>0.00</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>101348.88</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>15647311</td>\n",
       "      <td>Hill</td>\n",
       "      <td>608</td>\n",
       "      <td>Spain</td>\n",
       "      <td>Female</td>\n",
       "      <td>41</td>\n",
       "      <td>1</td>\n",
       "      <td>83807.86</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>112542.58</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>15619304</td>\n",
       "      <td>Onio</td>\n",
       "      <td>502</td>\n",
       "      <td>France</td>\n",
       "      <td>Female</td>\n",
       "      <td>42</td>\n",
       "      <td>8</td>\n",
       "      <td>159660.80</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>113931.57</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>15701354</td>\n",
       "      <td>Boni</td>\n",
       "      <td>699</td>\n",
       "      <td>France</td>\n",
       "      <td>Female</td>\n",
       "      <td>39</td>\n",
       "      <td>1</td>\n",
       "      <td>0.00</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>93826.63</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>15737888</td>\n",
       "      <td>Mitchell</td>\n",
       "      <td>850</td>\n",
       "      <td>Spain</td>\n",
       "      <td>Female</td>\n",
       "      <td>43</td>\n",
       "      <td>2</td>\n",
       "      <td>125510.82</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>79084.10</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   RowNumber  CustomerId   Surname  CreditScore Geography  Gender  Age  \\\n",
       "0          1    15634602  Hargrave          619    France  Female   42   \n",
       "1          2    15647311      Hill          608     Spain  Female   41   \n",
       "2          3    15619304      Onio          502    France  Female   42   \n",
       "3          4    15701354      Boni          699    France  Female   39   \n",
       "4          5    15737888  Mitchell          850     Spain  Female   43   \n",
       "\n",
       "   Tenure    Balance  NumOfProducts  HasCrCard  IsActiveMember  \\\n",
       "0       2       0.00              1          1               1   \n",
       "1       1   83807.86              1          0               1   \n",
       "2       8  159660.80              3          1               0   \n",
       "3       1       0.00              2          0               0   \n",
       "4       2  125510.82              1          1               1   \n",
       "\n",
       "   EstimatedSalary  Exited  \n",
       "0        101348.88       1  \n",
       "1        112542.58       0  \n",
       "2        113931.57       1  \n",
       "3         93826.63       0  \n",
       "4         79084.10       0  "
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "banking_churn.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "9e7aa99f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 10000 entries, 0 to 9999\n",
      "Data columns (total 14 columns):\n",
      " #   Column           Non-Null Count  Dtype  \n",
      "---  ------           --------------  -----  \n",
      " 0   RowNumber        10000 non-null  int64  \n",
      " 1   CustomerId       10000 non-null  int64  \n",
      " 2   Surname          10000 non-null  object \n",
      " 3   CreditScore      10000 non-null  int64  \n",
      " 4   Geography        10000 non-null  object \n",
      " 5   Gender           10000 non-null  object \n",
      " 6   Age              10000 non-null  int64  \n",
      " 7   Tenure           10000 non-null  int64  \n",
      " 8   Balance          10000 non-null  float64\n",
      " 9   NumOfProducts    10000 non-null  int64  \n",
      " 10  HasCrCard        10000 non-null  int64  \n",
      " 11  IsActiveMember   10000 non-null  int64  \n",
      " 12  EstimatedSalary  10000 non-null  float64\n",
      " 13  Exited           10000 non-null  int64  \n",
      "dtypes: float64(2), int64(9), object(3)\n",
      "memory usage: 1.1+ MB\n"
     ]
    }
   ],
   "source": [
    "banking_churn.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "c8e65d15",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>RowNumber</th>\n",
       "      <th>CustomerId</th>\n",
       "      <th>CreditScore</th>\n",
       "      <th>Age</th>\n",
       "      <th>Tenure</th>\n",
       "      <th>Balance</th>\n",
       "      <th>NumOfProducts</th>\n",
       "      <th>HasCrCard</th>\n",
       "      <th>IsActiveMember</th>\n",
       "      <th>EstimatedSalary</th>\n",
       "      <th>Exited</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>10000.00000</td>\n",
       "      <td>1.000000e+04</td>\n",
       "      <td>10000.000000</td>\n",
       "      <td>10000.000000</td>\n",
       "      <td>10000.000000</td>\n",
       "      <td>10000.000000</td>\n",
       "      <td>10000.000000</td>\n",
       "      <td>10000.00000</td>\n",
       "      <td>10000.000000</td>\n",
       "      <td>10000.000000</td>\n",
       "      <td>10000.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>5000.50000</td>\n",
       "      <td>1.569094e+07</td>\n",
       "      <td>650.528800</td>\n",
       "      <td>38.921800</td>\n",
       "      <td>5.012800</td>\n",
       "      <td>76485.889288</td>\n",
       "      <td>1.530200</td>\n",
       "      <td>0.70550</td>\n",
       "      <td>0.515100</td>\n",
       "      <td>100090.239881</td>\n",
       "      <td>0.203700</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>2886.89568</td>\n",
       "      <td>7.193619e+04</td>\n",
       "      <td>96.653299</td>\n",
       "      <td>10.487806</td>\n",
       "      <td>2.892174</td>\n",
       "      <td>62397.405202</td>\n",
       "      <td>0.581654</td>\n",
       "      <td>0.45584</td>\n",
       "      <td>0.499797</td>\n",
       "      <td>57510.492818</td>\n",
       "      <td>0.402769</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>1.00000</td>\n",
       "      <td>1.556570e+07</td>\n",
       "      <td>350.000000</td>\n",
       "      <td>18.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.00000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>11.580000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>2500.75000</td>\n",
       "      <td>1.562853e+07</td>\n",
       "      <td>584.000000</td>\n",
       "      <td>32.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.00000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>51002.110000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>5000.50000</td>\n",
       "      <td>1.569074e+07</td>\n",
       "      <td>652.000000</td>\n",
       "      <td>37.000000</td>\n",
       "      <td>5.000000</td>\n",
       "      <td>97198.540000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.00000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>100193.915000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>7500.25000</td>\n",
       "      <td>1.575323e+07</td>\n",
       "      <td>718.000000</td>\n",
       "      <td>44.000000</td>\n",
       "      <td>7.000000</td>\n",
       "      <td>127644.240000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>1.00000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>149388.247500</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>10000.00000</td>\n",
       "      <td>1.581569e+07</td>\n",
       "      <td>850.000000</td>\n",
       "      <td>92.000000</td>\n",
       "      <td>10.000000</td>\n",
       "      <td>250898.090000</td>\n",
       "      <td>4.000000</td>\n",
       "      <td>1.00000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>199992.480000</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "         RowNumber    CustomerId   CreditScore           Age        Tenure  \\\n",
       "count  10000.00000  1.000000e+04  10000.000000  10000.000000  10000.000000   \n",
       "mean    5000.50000  1.569094e+07    650.528800     38.921800      5.012800   \n",
       "std     2886.89568  7.193619e+04     96.653299     10.487806      2.892174   \n",
       "min        1.00000  1.556570e+07    350.000000     18.000000      0.000000   \n",
       "25%     2500.75000  1.562853e+07    584.000000     32.000000      3.000000   \n",
       "50%     5000.50000  1.569074e+07    652.000000     37.000000      5.000000   \n",
       "75%     7500.25000  1.575323e+07    718.000000     44.000000      7.000000   \n",
       "max    10000.00000  1.581569e+07    850.000000     92.000000     10.000000   \n",
       "\n",
       "             Balance  NumOfProducts    HasCrCard  IsActiveMember  \\\n",
       "count   10000.000000   10000.000000  10000.00000    10000.000000   \n",
       "mean    76485.889288       1.530200      0.70550        0.515100   \n",
       "std     62397.405202       0.581654      0.45584        0.499797   \n",
       "min         0.000000       1.000000      0.00000        0.000000   \n",
       "25%         0.000000       1.000000      0.00000        0.000000   \n",
       "50%     97198.540000       1.000000      1.00000        1.000000   \n",
       "75%    127644.240000       2.000000      1.00000        1.000000   \n",
       "max    250898.090000       4.000000      1.00000        1.000000   \n",
       "\n",
       "       EstimatedSalary        Exited  \n",
       "count     10000.000000  10000.000000  \n",
       "mean     100090.239881      0.203700  \n",
       "std       57510.492818      0.402769  \n",
       "min          11.580000      0.000000  \n",
       "25%       51002.110000      0.000000  \n",
       "50%      100193.915000      0.000000  \n",
       "75%      149388.247500      0.000000  \n",
       "max      199992.480000      1.000000  "
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "banking_churn.describe()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "81bea926",
   "metadata": {},
   "source": [
    "#### 1 Splitting the data into features"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "9ec457b2",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>RowNumber</th>\n",
       "      <th>CustomerId</th>\n",
       "      <th>Surname</th>\n",
       "      <th>CreditScore</th>\n",
       "      <th>Geography</th>\n",
       "      <th>Gender</th>\n",
       "      <th>Age</th>\n",
       "      <th>Tenure</th>\n",
       "      <th>Balance</th>\n",
       "      <th>NumOfProducts</th>\n",
       "      <th>HasCrCard</th>\n",
       "      <th>IsActiveMember</th>\n",
       "      <th>EstimatedSalary</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>15634602</td>\n",
       "      <td>Hargrave</td>\n",
       "      <td>619</td>\n",
       "      <td>France</td>\n",
       "      <td>Female</td>\n",
       "      <td>42</td>\n",
       "      <td>2</td>\n",
       "      <td>0.00</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>101348.88</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>15647311</td>\n",
       "      <td>Hill</td>\n",
       "      <td>608</td>\n",
       "      <td>Spain</td>\n",
       "      <td>Female</td>\n",
       "      <td>41</td>\n",
       "      <td>1</td>\n",
       "      <td>83807.86</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>112542.58</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>15619304</td>\n",
       "      <td>Onio</td>\n",
       "      <td>502</td>\n",
       "      <td>France</td>\n",
       "      <td>Female</td>\n",
       "      <td>42</td>\n",
       "      <td>8</td>\n",
       "      <td>159660.80</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>113931.57</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>15701354</td>\n",
       "      <td>Boni</td>\n",
       "      <td>699</td>\n",
       "      <td>France</td>\n",
       "      <td>Female</td>\n",
       "      <td>39</td>\n",
       "      <td>1</td>\n",
       "      <td>0.00</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>93826.63</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>15737888</td>\n",
       "      <td>Mitchell</td>\n",
       "      <td>850</td>\n",
       "      <td>Spain</td>\n",
       "      <td>Female</td>\n",
       "      <td>43</td>\n",
       "      <td>2</td>\n",
       "      <td>125510.82</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>79084.10</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   RowNumber  CustomerId   Surname  CreditScore Geography  Gender  Age  \\\n",
       "0          1    15634602  Hargrave          619    France  Female   42   \n",
       "1          2    15647311      Hill          608     Spain  Female   41   \n",
       "2          3    15619304      Onio          502    France  Female   42   \n",
       "3          4    15701354      Boni          699    France  Female   39   \n",
       "4          5    15737888  Mitchell          850     Spain  Female   43   \n",
       "\n",
       "   Tenure    Balance  NumOfProducts  HasCrCard  IsActiveMember  \\\n",
       "0       2       0.00              1          1               1   \n",
       "1       1   83807.86              1          0               1   \n",
       "2       8  159660.80              3          1               0   \n",
       "3       1       0.00              2          0               0   \n",
       "4       2  125510.82              1          1               1   \n",
       "\n",
       "   EstimatedSalary  \n",
       "0        101348.88  \n",
       "1        112542.58  \n",
       "2        113931.57  \n",
       "3         93826.63  \n",
       "4         79084.10  "
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x = banking_churn.drop(\"Exited\", axis=1)\n",
    "x.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "c0d50811",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0    1\n",
       "1    0\n",
       "2    1\n",
       "3    0\n",
       "4    0\n",
       "Name: Exited, dtype: int64"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y = banking_churn[\"Exited\"]\n",
    "y.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 90,
   "id": "248beeec",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "((7000, 13), (3000, 13), (7000,), (3000,))"
      ]
     },
     "execution_count": 90,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Splitting the data into training and test sets\n",
    "from sklearn.model_selection import train_test_split\n",
    "x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=42, stratify=y) \n",
    "\n",
    "x_train.shape, x_test.shape, y_train.shape, y_test.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 91,
   "id": "373362f7",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>RowNumber</th>\n",
       "      <th>CustomerId</th>\n",
       "      <th>Surname</th>\n",
       "      <th>CreditScore</th>\n",
       "      <th>Geography</th>\n",
       "      <th>Gender</th>\n",
       "      <th>Age</th>\n",
       "      <th>Tenure</th>\n",
       "      <th>Balance</th>\n",
       "      <th>NumOfProducts</th>\n",
       "      <th>HasCrCard</th>\n",
       "      <th>IsActiveMember</th>\n",
       "      <th>EstimatedSalary</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>9136</th>\n",
       "      <td>9137</td>\n",
       "      <td>15688984</td>\n",
       "      <td>Belonwu</td>\n",
       "      <td>595</td>\n",
       "      <td>France</td>\n",
       "      <td>Male</td>\n",
       "      <td>20</td>\n",
       "      <td>4</td>\n",
       "      <td>95830.43</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>177738.98</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6410</th>\n",
       "      <td>6411</td>\n",
       "      <td>15762351</td>\n",
       "      <td>Chao</td>\n",
       "      <td>689</td>\n",
       "      <td>Spain</td>\n",
       "      <td>Female</td>\n",
       "      <td>63</td>\n",
       "      <td>1</td>\n",
       "      <td>0.00</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>186526.12</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2607</th>\n",
       "      <td>2608</td>\n",
       "      <td>15814209</td>\n",
       "      <td>Capon</td>\n",
       "      <td>814</td>\n",
       "      <td>France</td>\n",
       "      <td>Male</td>\n",
       "      <td>31</td>\n",
       "      <td>1</td>\n",
       "      <td>118870.92</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>101704.19</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3201</th>\n",
       "      <td>3202</td>\n",
       "      <td>15637593</td>\n",
       "      <td>Greco</td>\n",
       "      <td>722</td>\n",
       "      <td>France</td>\n",
       "      <td>Male</td>\n",
       "      <td>20</td>\n",
       "      <td>6</td>\n",
       "      <td>0.00</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>195486.28</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3330</th>\n",
       "      <td>3331</td>\n",
       "      <td>15657439</td>\n",
       "      <td>Chao</td>\n",
       "      <td>738</td>\n",
       "      <td>France</td>\n",
       "      <td>Male</td>\n",
       "      <td>18</td>\n",
       "      <td>4</td>\n",
       "      <td>0.00</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>47799.15</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "      RowNumber  CustomerId  Surname  CreditScore Geography  Gender  Age  \\\n",
       "9136       9137    15688984  Belonwu          595    France    Male   20   \n",
       "6410       6411    15762351     Chao          689     Spain  Female   63   \n",
       "2607       2608    15814209    Capon          814    France    Male   31   \n",
       "3201       3202    15637593    Greco          722    France    Male   20   \n",
       "3330       3331    15657439     Chao          738    France    Male   18   \n",
       "\n",
       "      Tenure    Balance  NumOfProducts  HasCrCard  IsActiveMember  \\\n",
       "9136       4   95830.43              1          1               0   \n",
       "6410       1       0.00              2          1               1   \n",
       "2607       1  118870.92              1          1               0   \n",
       "3201       6       0.00              2          1               0   \n",
       "3330       4       0.00              2          1               1   \n",
       "\n",
       "      EstimatedSalary  \n",
       "9136        177738.98  \n",
       "6410        186526.12  \n",
       "2607        101704.19  \n",
       "3201        195486.28  \n",
       "3330         47799.15  "
      ]
     },
     "execution_count": 91,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x_train.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 92,
   "id": "eb092a9d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "9136    0\n",
       "6410    0\n",
       "2607    0\n",
       "3201    0\n",
       "3330    0\n",
       "       ..\n",
       "1467    0\n",
       "4644    0\n",
       "8942    0\n",
       "2935    0\n",
       "6206    0\n",
       "Name: Exited, Length: 7000, dtype: int64"
      ]
     },
     "execution_count": 92,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_train"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 93,
   "id": "273a2437",
   "metadata": {},
   "outputs": [
    {
     "ename": "ValueError",
     "evalue": "could not convert string to float: 'Belonwu'",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mValueError\u001b[0m                                Traceback (most recent call last)",
      "\u001b[0;32m/var/folders/gd/j9z6slqx4yn9_2s48t3qn3nm0000gn/T/ipykernel_60089/1962277796.py\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m      3\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      4\u001b[0m \u001b[0mmodel\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mRandomForestRegressor\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 5\u001b[0;31m \u001b[0mmodel\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mfit\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mx_train\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0my_train\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      6\u001b[0m \u001b[0mmodel\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mscore\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mx_test\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0my_test\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/opt/anaconda3/lib/python3.9/site-packages/sklearn/ensemble/_forest.py\u001b[0m in \u001b[0;36mfit\u001b[0;34m(self, X, y, sample_weight)\u001b[0m\n\u001b[1;32m    302\u001b[0m                 \u001b[0;34m\"sparse multilabel-indicator for y is not supported.\"\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    303\u001b[0m             )\n\u001b[0;32m--> 304\u001b[0;31m         X, y = self._validate_data(X, y, multi_output=True,\n\u001b[0m\u001b[1;32m    305\u001b[0m                                    accept_sparse=\"csc\", dtype=DTYPE)\n\u001b[1;32m    306\u001b[0m         \u001b[0;32mif\u001b[0m \u001b[0msample_weight\u001b[0m \u001b[0;32mis\u001b[0m \u001b[0;32mnot\u001b[0m \u001b[0;32mNone\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/opt/anaconda3/lib/python3.9/site-packages/sklearn/base.py\u001b[0m in \u001b[0;36m_validate_data\u001b[0;34m(self, X, y, reset, validate_separately, **check_params)\u001b[0m\n\u001b[1;32m    431\u001b[0m                 \u001b[0my\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mcheck_array\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0my\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mcheck_y_params\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    432\u001b[0m             \u001b[0;32melse\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 433\u001b[0;31m                 \u001b[0mX\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0my\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mcheck_X_y\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mX\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0my\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mcheck_params\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    434\u001b[0m             \u001b[0mout\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mX\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0my\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    435\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/opt/anaconda3/lib/python3.9/site-packages/sklearn/utils/validation.py\u001b[0m in \u001b[0;36minner_f\u001b[0;34m(*args, **kwargs)\u001b[0m\n\u001b[1;32m     61\u001b[0m             \u001b[0mextra_args\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mlen\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0margs\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;34m-\u001b[0m \u001b[0mlen\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mall_args\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     62\u001b[0m             \u001b[0;32mif\u001b[0m \u001b[0mextra_args\u001b[0m \u001b[0;34m<=\u001b[0m \u001b[0;36m0\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 63\u001b[0;31m                 \u001b[0;32mreturn\u001b[0m \u001b[0mf\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m*\u001b[0m\u001b[0margs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     64\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     65\u001b[0m             \u001b[0;31m# extra_args > 0\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/opt/anaconda3/lib/python3.9/site-packages/sklearn/utils/validation.py\u001b[0m in \u001b[0;36mcheck_X_y\u001b[0;34m(X, y, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, multi_output, ensure_min_samples, ensure_min_features, y_numeric, estimator)\u001b[0m\n\u001b[1;32m    869\u001b[0m         \u001b[0;32mraise\u001b[0m \u001b[0mValueError\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\"y cannot be None\"\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    870\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 871\u001b[0;31m     X = check_array(X, accept_sparse=accept_sparse,\n\u001b[0m\u001b[1;32m    872\u001b[0m                     \u001b[0maccept_large_sparse\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0maccept_large_sparse\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    873\u001b[0m                     \u001b[0mdtype\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mdtype\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0morder\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0morder\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mcopy\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mcopy\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/opt/anaconda3/lib/python3.9/site-packages/sklearn/utils/validation.py\u001b[0m in \u001b[0;36minner_f\u001b[0;34m(*args, **kwargs)\u001b[0m\n\u001b[1;32m     61\u001b[0m             \u001b[0mextra_args\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mlen\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0margs\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;34m-\u001b[0m \u001b[0mlen\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mall_args\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     62\u001b[0m             \u001b[0;32mif\u001b[0m \u001b[0mextra_args\u001b[0m \u001b[0;34m<=\u001b[0m \u001b[0;36m0\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 63\u001b[0;31m                 \u001b[0;32mreturn\u001b[0m \u001b[0mf\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m*\u001b[0m\u001b[0margs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     64\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     65\u001b[0m             \u001b[0;31m# extra_args > 0\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/opt/anaconda3/lib/python3.9/site-packages/sklearn/utils/validation.py\u001b[0m in \u001b[0;36mcheck_array\u001b[0;34m(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, estimator)\u001b[0m\n\u001b[1;32m    671\u001b[0m                     \u001b[0marray\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0marray\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mastype\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mdtype\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mcasting\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m\"unsafe\"\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mcopy\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;32mFalse\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    672\u001b[0m                 \u001b[0;32melse\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 673\u001b[0;31m                     \u001b[0marray\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mnp\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0masarray\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0marray\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0morder\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0morder\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mdtype\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mdtype\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    674\u001b[0m             \u001b[0;32mexcept\u001b[0m \u001b[0mComplexWarning\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0mcomplex_warning\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    675\u001b[0m                 raise ValueError(\"Complex data not supported\\n\"\n",
      "\u001b[0;32m~/opt/anaconda3/lib/python3.9/site-packages/pandas/core/generic.py\u001b[0m in \u001b[0;36m__array__\u001b[0;34m(self, dtype)\u001b[0m\n\u001b[1;32m   2062\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   2063\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0m__array__\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mdtype\u001b[0m\u001b[0;34m:\u001b[0m \u001b[0mnpt\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mDTypeLike\u001b[0m \u001b[0;34m|\u001b[0m \u001b[0;32mNone\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;32mNone\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;34m->\u001b[0m \u001b[0mnp\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mndarray\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 2064\u001b[0;31m         \u001b[0;32mreturn\u001b[0m \u001b[0mnp\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0masarray\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_values\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mdtype\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mdtype\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   2065\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   2066\u001b[0m     def __array_wrap__(\n",
      "\u001b[0;31mValueError\u001b[0m: could not convert string to float: 'Belonwu'"
     ]
    }
   ],
   "source": [
    "# Try to predict with random forest on exited column (doesn't work)\n",
    "from sklearn.ensemble import RandomForestRegressor\n",
    "\n",
    "model = RandomForestRegressor()\n",
    "model.fit(x_train, y_train)\n",
    "model.score(x_test, y_test)"
   ]
  },
  {
   "cell_type": "code",
   #"execution_count": null,
   "id": "92a37ad8",
   "metadata": {},
   "outputs": [],
   "source": [
    "x=banking_churn.groupby(['EstimatedSalary','Exited']).Exited.count().unstack()\n",
    "x.plot(kind='bar',stacked=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 95,
   "id": "84f0cf5d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "RowNumber            int64\n",
       "CustomerId           int64\n",
       "Surname             object\n",
       "CreditScore          int64\n",
       "Geography           object\n",
       "Gender              object\n",
       "Age                  int64\n",
       "Tenure               int64\n",
       "Balance            float64\n",
       "NumOfProducts        int64\n",
       "HasCrCard            int64\n",
       "IsActiveMember       int64\n",
       "EstimatedSalary    float64\n",
       "dtype: object"
      ]
     },
     "execution_count": 95,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x_train.dtypes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 96,
   "id": "b739053d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<3000x1495 sparse matrix of type '<class 'numpy.float64'>'\n",
       "\twith 9000 stored elements in Compressed Sparse Row format>"
      ]
     },
     "execution_count": 96,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Turn the categories (Surname, Geography and Gender) into numbers\n",
    "from sklearn.preprocessing import OneHotEncoder\n",
    "from sklearn.compose import ColumnTransformer\n",
    "from sklearn.tree import DecisionTreeRegressor\n",
    "oneh = OneHotEncoder(handle_unknown=\"ignore\")\n",
    "\n",
    "\n",
    "categorical_features = [\"Surname\", \"Geography\", \"Gender\"]\n",
    "ohe = OneHotEncoder()\n",
    "transformer = ColumnTransformer(transformers = [(\"one_hot\", \n",
    "                                 ohe, \n",
    "                                 categorical_features)],\n",
    "                                 remainder=\"drop\")\n",
    "transformed_x_train = transformer.fit_transform(x_train)\n",
    "transformed_x_test = transformer.fit_transform(x_test)\n",
    "transformed_x_test"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 97,
   "id": "04e67386",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>0</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>(0, 179)\\t1.0\\n  (0, 2441)\\t1.0\\n  (0, 2445)...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>(0, 372)\\t1.0\\n  (0, 2443)\\t1.0\\n  (0, 2444)...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>(0, 330)\\t1.0\\n  (0, 2441)\\t1.0\\n  (0, 2445)...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>(0, 865)\\t1.0\\n  (0, 2441)\\t1.0\\n  (0, 2445)...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>(0, 372)\\t1.0\\n  (0, 2441)\\t1.0\\n  (0, 2445)...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                                   0\n",
       "0    (0, 179)\\t1.0\\n  (0, 2441)\\t1.0\\n  (0, 2445)...\n",
       "1    (0, 372)\\t1.0\\n  (0, 2443)\\t1.0\\n  (0, 2444)...\n",
       "2    (0, 330)\\t1.0\\n  (0, 2441)\\t1.0\\n  (0, 2445)...\n",
       "3    (0, 865)\\t1.0\\n  (0, 2441)\\t1.0\\n  (0, 2445)...\n",
       "4    (0, 372)\\t1.0\\n  (0, 2441)\\t1.0\\n  (0, 2445)..."
      ]
     },
     "execution_count": 97,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train = pd.DataFrame(transformed_x_train)\n",
    "#train.columns = x_train.columns\n",
    "train.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 102,
   "id": "624e4ece",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "RandomForestRegressor()"
      ]
     },
     "execution_count": 102,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Let's refit the model\n",
    "\n",
    "model.fit(transformed_x_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 100,
   "id": "d5dd7908",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.6103478815828409"
      ]
     },
     "execution_count": 100,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.score(transformed_x_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "ab69b544",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>RowNumber</th>\n",
       "      <th>CustomerId</th>\n",
       "      <th>Surname</th>\n",
       "      <th>CreditScore</th>\n",
       "      <th>Geography</th>\n",
       "      <th>Gender</th>\n",
       "      <th>Age</th>\n",
       "      <th>Tenure</th>\n",
       "      <th>Balance</th>\n",
       "      <th>NumOfProducts</th>\n",
       "      <th>HasCrCard</th>\n",
       "      <th>IsActiveMember</th>\n",
       "      <th>EstimatedSalary</th>\n",
       "      <th>Exited</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>15634602</td>\n",
       "      <td>Hargrave</td>\n",
       "      <td>619</td>\n",
       "      <td>France</td>\n",
       "      <td>Female</td>\n",
       "      <td>42</td>\n",
       "      <td>2</td>\n",
       "      <td>0.00</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>101348.88</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>15647311</td>\n",
       "      <td>Hill</td>\n",
       "      <td>608</td>\n",
       "      <td>Spain</td>\n",
       "      <td>Female</td>\n",
       "      <td>41</td>\n",
       "      <td>1</td>\n",
       "      <td>83807.86</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>112542.58</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>15619304</td>\n",
       "      <td>Onio</td>\n",
       "      <td>502</td>\n",
       "      <td>France</td>\n",
       "      <td>Female</td>\n",
       "      <td>42</td>\n",
       "      <td>8</td>\n",
       "      <td>159660.80</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>113931.57</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>15701354</td>\n",
       "      <td>Boni</td>\n",
       "      <td>699</td>\n",
       "      <td>France</td>\n",
       "      <td>Female</td>\n",
       "      <td>39</td>\n",
       "      <td>1</td>\n",
       "      <td>0.00</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>93826.63</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   RowNumber  CustomerId   Surname  CreditScore Geography  Gender  Age  \\\n",
       "0          1    15634602  Hargrave          619    France  Female   42   \n",
       "1          2    15647311      Hill          608     Spain  Female   41   \n",
       "2          3    15619304      Onio          502    France  Female   42   \n",
       "3          4    15701354      Boni          699    France  Female   39   \n",
       "\n",
       "   Tenure    Balance  NumOfProducts  HasCrCard  IsActiveMember  \\\n",
       "0       2       0.00              1          1               1   \n",
       "1       1   83807.86              1          0               1   \n",
       "2       8  159660.80              3          1               0   \n",
       "3       1       0.00              2          0               0   \n",
       "\n",
       "   EstimatedSalary  Exited  \n",
       "0        101348.88       1  \n",
       "1        112542.58       0  \n",
       "2        113931.57       1  \n",
       "3         93826.63       0  "
      ]
     },
     "execution_count": 37,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Import banking_churn dataframe with missing values\n",
    "banking_churn_missing = pd.read_csv(\"https://raw.githubusercontent.com/Livuza/ADS-April-2021/main/Assignments/Assignment%202/banking_churn.csv\")\n",
    "banking_churn_missing.head(4)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "bae97d98",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "RowNumber          0\n",
       "CustomerId         0\n",
       "Surname            0\n",
       "CreditScore        0\n",
       "Geography          0\n",
       "Gender             0\n",
       "Age                0\n",
       "Tenure             0\n",
       "Balance            0\n",
       "NumOfProducts      0\n",
       "HasCrCard          0\n",
       "IsActiveMember     0\n",
       "EstimatedSalary    0\n",
       "Exited             0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 38,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "banking_churn_missing.isna().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "4f598d56",
   "metadata": {},
   "outputs": [],
   "source": [
    "banking_churn_missing.dropna(subset=['Exited'], inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "7b830d46",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "RowNumber          0\n",
       "CustomerId         0\n",
       "Surname            0\n",
       "CreditScore        0\n",
       "Geography          0\n",
       "Gender             0\n",
       "Age                0\n",
       "Tenure             0\n",
       "Balance            0\n",
       "NumOfProducts      0\n",
       "HasCrCard          0\n",
       "IsActiveMember     0\n",
       "EstimatedSalary    0\n",
       "Exited             0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 40,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "banking_churn_missing.isna().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 104,
   "id": "983d162d",
   "metadata": {},
   "outputs": [],
   "source": [
    "x = banking_churn_missing.drop(\"Exited\", axis=1)\n",
    "y = banking_churn_missing[\"Exited\"]\n",
    "# Split data into train and test\n",
    "x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=21)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "id": "69359f6d",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Fill the \"Geography\" column\n",
    "x_train[\"Geography\"].fillna(\"missing\", inplace=True)\n",
    "x_test[\"Geography\"].fillna(\"missing\", inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "id": "4b504845",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "RowNumber          0\n",
       "CustomerId         0\n",
       "Surname            0\n",
       "CreditScore        0\n",
       "Geography          0\n",
       "Gender             0\n",
       "Age                0\n",
       "Tenure             0\n",
       "Balance            0\n",
       "NumOfProducts      0\n",
       "HasCrCard          0\n",
       "IsActiveMember     0\n",
       "EstimatedSalary    0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 45,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x_train.isna().sum()\n",
    "x_test.isna().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "id": "93ec3245",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0"
      ]
     },
     "execution_count": 46,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_test.isna().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "id": "c16d59fa",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0"
      ]
     },
     "execution_count": 47,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_train.isna().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 110,
   "id": "5bd4895c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[1., 0., 0., 1., 0.],\n",
       "       [0., 0., 1., 1., 0.],\n",
       "       [1., 0., 0., 0., 1.],\n",
       "       ...,\n",
       "       [0., 0., 1., 0., 1.],\n",
       "       [0., 1., 0., 0., 1.],\n",
       "       [0., 1., 0., 0., 1.]])"
      ]
     },
     "execution_count": 110,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "categorical_features = [\"Geography\", \"Gender\"]\n",
    "one_hot = OneHotEncoder()\n",
    "transformer = ColumnTransformer(transformers = [(\"one_hot\", \n",
    "                                 one_hot, \n",
    "                                 categorical_features)],\n",
    "                                 remainder=\"drop\")\n",
    "transformed_x_train = transformer.fit_transform(x_train)\n",
    "transformed_x_test = transformer.fit_transform(x_test)\n",
    "transformed_x_test"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "id": "ef64ffdb",
   "metadata": {},
   "outputs": [],
   "source": [
    "x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "id": "14b76aef",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.impute import SimpleImputer"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 112,
   "id": "181ed22a",
   "metadata": {},
   "outputs": [],
   "source": [
    "categorical_features = [\"Geography\", \"Gender\"]\n",
    "numerical_feature = [\"NumOfProducts\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 113,
   "id": "f18a696a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([['France', 'Male', 2.0],\n",
       "       ['France', 'Female', 1.0],\n",
       "       ['France', 'Female', 1.0],\n",
       "       ...,\n",
       "       ['Spain', 'Male', 1.0],\n",
       "       ['Germany', 'Male', 1.0],\n",
       "       ['Germany', 'Female', 2.0]], dtype=object)"
      ]
     },
     "execution_count": 113,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "imputer = ColumnTransformer(transformers = [\n",
    "    (\"cat_imputer\", cat_imputer, categorical_features),\n",
    "    (\"num_imputer\", num_imputer, numerical_feature)])\n",
    "\n",
    "# Fill train and test values separately\n",
    "filled_x_train = imputer.fit_transform(x_train)\n",
    "filled_x_test = imputer.transform(x_test)\n",
    "\n",
    "# Check filled X_train\n",
    "filled_x_train"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 119,
   "id": "b87a9a84",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Geography        0\n",
       "Gender           0\n",
       "NumOfProducts    0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 119,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "banking_churn_filled_train = pd.DataFrame(filled_x_train, columns=[\"Geography\", \"Gender\", \"NumOfProducts\"])\n",
    "                                      \n",
    "\n",
    "banking_churn_filled_test = pd.DataFrame(filled_x_test, columns=[\"Geography\", \"Gender\", \"NumOfProducts\"])\n",
    "                                     \n",
    "\n",
    "# Check missing data in training set\n",
    "banking_churn_filled_train.isna().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 120,
   "id": "2078776c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Geography</th>\n",
       "      <th>Gender</th>\n",
       "      <th>NumOfProducts</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>France</td>\n",
       "      <td>Male</td>\n",
       "      <td>2.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>France</td>\n",
       "      <td>Female</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>France</td>\n",
       "      <td>Female</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Spain</td>\n",
       "      <td>Male</td>\n",
       "      <td>2.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>France</td>\n",
       "      <td>Female</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "  Geography  Gender NumOfProducts\n",
       "0    France    Male           2.0\n",
       "1    France  Female           1.0\n",
       "2    France  Female           1.0\n",
       "3     Spain    Male           2.0\n",
       "4    France  Female           1.0"
      ]
     },
     "execution_count": 120,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "banking_churn_filled_train.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "764ee3b3",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
