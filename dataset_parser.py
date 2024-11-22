import collections # set
import json
import ast
import pickle # save parsed table to file

"""
Classes:
    - SetOfSeenAttributes
    - DatasetParser
    - ParsedDatasetRow
"""

PICKLE_FILE_PATH = 'saved_dictionary.pkl'

class SetOfSeenAttributes():
    """
    A "set" that counts the number of times a duplicate item has tried to be inserted
    """
    def __init__(self):
        self.set = {} # dict

    def add(self, item):
        if item in self.set:
            self.set[item] = self.set[item] + 1
        else:
            self.set[item] = 1
    
    def print(self):
        sorted_dict = dict(sorted(self.set.items(), key=lambda item: item))
        for item in sorted_dict:
            print(f"{item}, {sorted_dict[item]}")

class DatasetParser():
    def __init__(self):
        self.table = []
        self.set_of_seen_attributes = SetOfSeenAttributes()

    def format_df_chunk(self, chunk):
        for i in range(len(chunk)):
            error = None
            dict_1 = None
            details = None
            features = None
            categories = None
            description = None
            try:
                cols = ['title'
                        , 'main_category'
                        , 'average_rating'
                        , 'rating_number'
                        , 'price'
                        , 'store'
                        , 'parent_asin'
                        , 'bought_together'
                        , 'subtitle'
                        , 'author']
                dict_1 = chunk[cols].iloc[i].fillna('').to_dict()
                details = self._convert_col_to_dict(chunk['details'].iloc[i])
                features = self._convert_col_to_list(chunk['features'].iloc[i])
                categories = {'categories': self._convert_col_to_list(chunk['categories'].iloc[i])}
                description = {'description': self._convert_col_to_list(chunk['description'].iloc[i])}
            except Exception as e:
                error = f"when parsing details {details}" if details is None else f"when parsing features {features}"

                print(f"ERROR: {e}\n{error}\n")
                break
            finally:
                # load everything into one dictionary
                parsed_row = ParsedDatasetRow(
                    raw_fields = [dict_1, details, features, categories, description]
                    ,set_of_seen_attributes = self.set_of_seen_attributes
                    )
                self.table.append(parsed_row)
                if error:
                    print(f"i: {i}")
                    parsed_row.print()
        print(f"loaded {len(self.table)}/{len(chunk)} rows into table")

    def print_seen_attributes(self):
        print("\nSeen attributes:")
        self.set_of_seen_attributes.print()
    
    def print_key_values(self, k):
        """
        Iterates through the data to print all the unique values for the passed key k and how often they occur.
        """
        values = dict()
        for row in self.table:
            v = row.fields.get(k)
            if v:
                for item in v:
                    num_item = (values.get(item) or 0) + 1 # add one to the count of values, or default to 1
                    values.update({item: num_item})
        print(f'Key: {k}\nValues: {values}')
        return values

    def write_table_to_file(self, selected_cols=[], append=True):
        if not selected_cols:
            selected_cols = range(0, len(self.table))
        # write to file
        with open("test_cleaned_table_output.txt", "a") as f:
            if not append:
                f.truncate(0) # clear the file
            for i in selected_cols:    
                f.write(f"{i}\n")
                row = self.table[i]
                for r in row.fields:
                    f.write(f"{r}: {row.fields[r]}")
                    f.write("\n")

    def save_to_file(self, filename=PICKLE_FILE_PATH):
        """
        Saves the dictionary as a pickle file to be loaded later
        filename: .pkl ender
        """
        # wb+ to create the file if it doesn't exist, then write into it
        with open(filename, 'wb+') as f:
            pickle.dump(self.table, f)
        print(f"Saved parsed dataset to file: {filename}")

    def load_from_file(self, filename=PICKLE_FILE_PATH):
        """
        Loads pickle file of saved dataset from file.
        """
        with open(filename, 'rb') as f:
            self.table= pickle.load(f)
        print(f"Loaded parsed dataset from file: {filename}")


    def _convert_col_to_dict(self, col):
        try:
            # handle single quotes, including cases like 'Skully's Ctz Beard Oil'
            return json.loads(json.dumps(ast.literal_eval(col)))
        except Exception as e:
            print(f"ERROR in _convert_col_to_dict: {e}\n{type(col)}, {col}")
            raise e

    def _convert_to_tuples(self, node):
        if isinstance(node, ast.List):
            return tuple(self._convert_to_tuples(element) for element in node.elts)
        elif isinstance(node, ast.Constant):
            return node.value
        else:
            print(f"type{node}")
            return node

    def _convert_col_to_list(self, col):
        col_list = None
        try:
            if type(col) == list:
                col_list = col
            elif type(col) == str:
                col_list = self._convert_to_tuples(ast.parse(col, mode='eval').body)
            else:
                print(f"Unknown type in convert_col_to_list: {type(col)}, {col}")
                col_list =col
        except Exception as e:
            print(f"ERROR in convert_col_to_list: {e}\n{type(col)}, {col}")
            col_list = None
        return col_list


class ParsedDatasetRow():
    def __init__(self, raw_fields, set_of_seen_attributes):
        self.set_of_seen_attributes = set_of_seen_attributes # must go before combine_fields
        self.fields = self.combine_fields(raw_fields)        

    def combine_fields(self, raw_fields):
        """
        fields: a list of dictionaries or lists to combine into a dictionary
        returns: a dictionary with unique values for each key
        """
        self.super_dict = collections.defaultdict(set) # no duplicates in values of a key
        
        for d in raw_fields:
            if type(d) == dict:
                self._process_dict(d)
            elif type(d) == list:
                try:
                    k = '???'
                    for v in d:
                        v = self._clean_value(self.v_item)
                        self.super_dict[k] = v
                except Exception as e:
                    print(f"ERROR in combine_fields when adding list to super_dict: {e}")
                    print(f"k: {k}, v: {type(v)} {v}")
                    raise e
        return self.super_dict

    def print(self):
        for k,v in self.fields.items():
            if k == 'features':
                print(f"    {k}")
                for feature in v:
                    print(f"        {feature}")
            else:
                print(f"    {k}: {v}")

    def _process_dict(self, d):
        try:
            for k, v in d.items():
                k, v = self._parse_features(k, v)
                self.set_of_seen_attributes.add(k)
                if type(v) == list:
                    for v_item in v:
                        if type(v_item) == str:
                            for i in v_item.split(','):
                                i = self._clean_value(i)
                                if i != '':
                                    self.super_dict[k].add(i)
                        else: # int, float...
                            self.super_dict[k].add(v_item)
                elif type(v) == dict:
                    self._process_dict(v)
                else:
                    v = self._clean_value(v)
                    self.super_dict[k].add(v)
        except Exception as e:
            print(f"ERROR in combine_fields when adding dict to super_dict: {e}")
            print(f"k: {k}, v: {type(v)} {v}")
            raise e

    def _parse_features(self, k, v):

        # Standardize feature names
        k = k.lower() # lowercase for processing
        k = k.replace('  ', ' ').replace(' ','_').replace('?','').replace(':','')

        # discard these key/value pairs
        #   pricing: 'The strikethrough price is the List Price. Savings represents a discount off the List Price.'
        if k in ['number_of_items', 'part_number', 'pricing', 'return_policy', 'terms_of_use']:
            return '', ''

        # Clean up wrong entries where the value was entered as part of the key
        #   e.g., 'surface_recommendation_stone' : '' -> 'surface_recommendation': 'stone'
        for item in ['installation_type', 'mounting_type', 'number_of_compartments'
                     , 'shape', 'surface_recommendation', 'item_weight', 'vehicle_service_type'
                     , 'style', 'wattage', 'voltage', 'load_index', 'climate_pledge_friendly'
                     , 'tire_aspect_ratio', 'heating_method', 'exterior_finish'
                     , 'number_of_drawers', 'number_of_doors', 'number_of_blades'
                     , 'battery_cell_composition', 'occasion', 'power_source', 'rim_size'
                     , 'age_range_(description)', 'maximum_weight_recommendation', 'capacity'
                     , 'form_factor', 'material'
                     # 'type', 
                    ]:
            if item in k:
                if k not in ['load_index_rating', 'type_of_bulb', 'typbe_of_item', 'typewriters']:
                    k, v = self._split_v_from_k(k, v, item)
        
        # Catch all for entries using the 'features' column as an item descriptor
        #  e.g., 'kids' toys and accessories': '' -> 'possible_category': 'kids' toys and accessories'
        if v == '' and k not in ['author', 'bought_together', 'subtitle', 'main_category', 'price']:
            v = k
            k = 'possible_category'
        # Catch all for entries in the 'features' column that may be category rankings instead
        #  e.g., musical_instruments: {86657} or saxophone_cleaning_&_care: {162}
        # match existing 'best_sellers_rank' feature
        if type(v) == int or (type(v) == str and v.isdigit()):
            #if k not in ['average_rating', 'rating_number', 'model_year']:
            unmatched = True
            for match in ['rating', 'year', 'total', 'number']:
                if match in k:
                    unmatched = False
            if unmatched:
                v = (k, v)
                k = 'best_sellers_rank'
        
        # Combine duplicate features under different names, spellings, or plurality
        ['batteries', 'batteries required', 'batteries included']
        if k == 'brand_name':
            k = 'brand'
        elif k == 'color_name':
            k = 'color'
        elif k == 'country/region_of_origin':
            k = 'country_of_origin'
        elif k == 'finish_types':
            k = 'finish_type'
        elif k == 'item_dimensions_lxwxh':
            k = 'item_dimensions'
        elif k == 'item_package_dimensions_l_x_w_x_h':
            k = 'package_dimensions'
        elif (k in ['number_of_pieces'
                , 'number_of_items'
                , 'unit_count']):
            k = 'pieces'
        elif k == 'package_information':
            k = 'package_type'
        elif k == 'connectivity_technology':
            k = 'connectivity_technologies'
        elif k == 'special_feature':
            k = 'special_features'
        elif k in ['use_for'
                ,'specific_uses_for_product'
                ,'recommended_uses_for_product']:
            k = 'usage'
        elif k in ['our_recommended_age'
                , 'manufacturer_recommended_age'
                , 'age_range_(description)'
                , 'rated']:
            k = 'recommended_age'
        elif k in ['suggested_users']:
            k = 'suggested_users'

        # Clean up all definitions of materials under one large 'material' ground
        #   e.g, tire material, hat material, fabric material, car material, paint material...
        if 'material' in k:
            k = 'material'
        
        # Clean up values that have been saved as a single string
        #   e.g., 'product_benefits':'clean, smooth, comfortable' -> 'product_benefits': ['clean', 'smooth', 'comfortable']
        if k in ['package_dimensions'
                 ,'product_dimensions'
                 ,'item_dimensions']:
            return k, v.split(';')
        elif k in ['product_benefits']:
            return k, v.split(';')
        elif k == 'description':
            return k, ' '.join(v) # combine array of strings into one string
        elif k == 'upc':
            # clean up malformed UPC where other features have been included after the UPC number
            #   e.g., 'UPC': '00000000, feature: value, feature 2: value 2'
            return k, v.split(',')[0]
        return k, v
    
    def _split_v_from_k(self, k, v, name):
        """
        Handles {k,v} cases like 
        { 'Shape Heart', '' }
        { 'Mounting Type Freestanding', ''}
        { 'Number of Compartments 3', '' }

        name: the root in lowercase, e.g., 'shape', 'mounting type', 'number of components'
        """
        k_split = k.split(name)
        k = name
        if len(k_split) > 1:
            new_v = []
            new_v.append(v)

            # split cases like 'style': 'modern,vintage,vintage_style'
            post_split = k_split[1:]
            for i in post_split:
                for val in i.split(','):
                    new_v.append(val.strip())

            v = new_v
        return k, v

    def _clean_value(self, v):
        if type(v) == str:
            v = v.replace('_',' ').strip() # handle cases like '_electronics'
            v.replace('\xa0',' ')
        return v

   