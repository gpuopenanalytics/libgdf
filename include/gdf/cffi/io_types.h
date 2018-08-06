/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * API
 *
 *
 * gdf_column ** read_csv(csv_read_arg *args,  int *num_col_out, gdf_error *error_out);
 *
 */


typedef struct {

	/*
	 * Output Arguments - space created in reader.
	 */
	int				num_cols;							// Out: number of columns
	int				num_rows;							// Out: number of rows
	gdf_column		**data;								// Out: the gdf_columns


	/*
	 * Input arguments
	 */
	const char		*file_path			= NULL;			// file location to read from	- if the file is compressed, it needs proper file extensions {.gz}
	const char		*buffer				= NULL;			// process data from a buffer,  pointer to Host memory
	const char 		*object				= NULL;			// this is a URL path

	const char		lineterminator		= '\n';			// can change the end of line character
	const char		delimiter			= ',';			// also called 'sep'  this is the field separator
	const bool 		delim_whitespace	= false;		// use white space as the delimiter
	const bool		skipinitialspace	= false;		// Skip spaces after delimiter

	const int		nrows				= -1;			// number of rows to read,  -1 indicates all

	const int		**header			= NULL;			// array of int: 	list of row numbers to use to use as column names,  zero based counting
	const char		**names				= NULL;			// array of char *  Ordered List of column names to use.   names cannot be used with header
	const char		*dtype				= NULL;			// array of char *	Ordered List of data types as strings

	const int		**index_col			= NULL;			// array of int:	Column to use as the row labels of the DataFrame.

	const int		**usecols			= NULL;			// array of int:	Return a subset of the columns.  CSV reader will only process those columns,  another read is needed to get full data

	const int		**skiprows			= NULL;			// array of int		number of rows at the start of the files to skip
	const int		**skipfooter		= NULL;			// array of int		number of rows at the bottom of the file to skip - counting is backwards from end, 0 = last line

	const bool		skip_blank_lines	= true;			// whether or not to ignore blank lines

	const char		*prefix				= NULL;			// if there is no header or names, attend this to the column ID as the name
	const bool		mangle_dupe_cols	= false;		// if true: duplicate columns will be specified as ‘X’, ‘X.1’, …’X.N’, rather than ‘X’…’X’  - CSV reader does not utilize column name for processing

	const char		**true_values		= NULL;			// array of char *	what should be considered as True - each char string contains {col ID, true value, ...} - this will allow multiple true values to be specified over multiple columns
	const char		**false_values		= NULL;			// array of char *	what should be considered as True - each char string contains {col ID, false value, ...} - this will allow multiple false values to be specified over multiple columns
	const char		**na_values			= NULL;			// array of char *	what should be considered as True - each char string contains {col ID, na value, ...} - this will allow multiple na values to be specified over multiple columns

	const bool		parse_dates			= true;			// parse date field into date32 or date64.  If false then date fields are saved as a string. Specifying a date dtype overrides this
	const bool		infer_datetime_format = true;		// try and determine the date format
	const bool		dayfirst			= false;		// is the first value the day?  DD/MM  versus MM/DD

	const char		*compression		= NULL;			// specify the type of compression

	const char 		*thousands			= NULL;			// single character		a separate within numeric data  - if this matches the delimiter then system will return NULL

	const char		decimal				= '.';			// the decimal point character

	const char		*quotechar			= NULL;			// single character 	character to use as a quote
	const bool		doublequote			= true;			// whether to treat two quotes as a quote in a string or empty

	const char		*escapechar			= NULL;			// single char	- char used as the escape character

	const char		*comment			= NULL;			// single char	- treat line or remainder of line as an unprocessed comment

	const char		*encoding			= NULL;			// the data encoding, NULL = UTF-8

} csv_read_arg;



/*
 * NOT USED
 *
 * squeeze			- data is always returned as a gdf_column array
 * engine			- this is the only engine
 * keep_default_na  - this has no meaning since the field is marked invalid and the value not seen
 * na_filter		- empty fields are automatically tagged as invalid
 * verbose
 * keep_date_col	- will not maintain raw data
 * date_parser		- there is only this parser
 * float_precision	- there is only one converter that will cover all specified values
 * quoting			- this is for out
 * dialect			- not used
 *
 */




