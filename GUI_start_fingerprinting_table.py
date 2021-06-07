from __future__ import print_function
import timeit
import GUI_config_fingerprinting_table as config_fingerprinting
import GUI_config_MultipleTabs as config_objects
import config_DWRegal as config
from multiprocessing import freeze_support
from GUI_class_generate_fingerprinting_table import GenerateFingerprintingTable


class GUI_start_fingerprinting_table():
    def __init__(self,save_table_name,look_up_table_path):
        self.save_table_name= save_table_name
        self.look_up_table_path =look_up_table_path

    def get_path_and_save_file_name(self):
        save_table_name= self.save_table_name
        look_up_table_path = self.look_up_table_path
        return save_table_name,look_up_table_path

    def main_sft(self):
        freeze_support()

        # path =r'C:\Users\Dauser\Desktop\dauserml_Messungen_2020_07_22\Referenz_alfred\Used_Data\x_y_z_position_look_up_table_compare.npy'
        # print('\nApplication: ' + config_fingerprinting.app)
        gcft = config_fingerprinting.GUI_config_fingerprinting_table()
        app, numberOfCores, author, tableFormat, positions, angles, additional_info = gcft.main(path=self.look_up_table_path)
        generateTable = GenerateFingerprintingTable(config_objects.exciter, config_objects.coil,
                                                    config.antennaList,
                                                    config_objects.frequency, config_objects.coilResistance,
                                                    positions, angles,
                                                    config_objects.objectType, numberOfCores,
                                                    app, author,
                                                    additional_info)
        start = timeit.default_timer()
        generateTable.calculate_table()
        stop = timeit.default_timer()
        print('Runtime:  {0} min'.format((stop - start) / 60))
        generateTable.save_results(path=self.look_up_table_path, tableName=self.save_table_name,
                                   tableFormat=tableFormat)
        start = timeit.default_timer()

        stop = timeit.default_timer()
        print('Time taken to save the table:  {0} min'.format((stop - start) / 60))
# This is script does the following:
#--> Create a GernerateFingerprintingTable Object
#--> Passing the configurations from both GUI_config_fingerprinting_table and
# GUI_config_MultipleTabs to that object
#--> Calculate the fingerprinting table using calculate_table() method
#--> Save the calculated table using save_results() method
if __name__ == '__main__':
    freeze_support()

    #path =r'C:\Users\Dauser\Desktop\dauserml_Messungen_2020_07_22\Referenz_alfred\Used_Data\x_y_z_position_look_up_table_compare.npy'
    #print('\nApplication: ' + config_fingerprinting.app)
    save_table_name,look_up_table_path =GUI_start_fingerprinting_table.get_path_and_save_file_name()
    gcft = config_fingerprinting.GUI_config_fingerprinting_table()
    app,numberOfCores,author,tableFormat,positions,angles,additional_info = gcft.main(path=save_table_name)
    generateTable = GenerateFingerprintingTable(config_objects.exciter, config_objects.coil,
                                                config.antennaList,
                                                config_objects.frequency, config_objects.coilResistance,
                                                positions, angles,
                                                config_objects.objectType,numberOfCores,
                                                app, author,
                                                additional_info)
    start = timeit.default_timer()
    generateTable.calculate_table()
    stop = timeit.default_timer()
    print('Runtime:  {0} min'.format((stop - start) / 60))
    gsft = GUI_start_fingerprinting_table(save_table_name=save_table_name,look_up_table_path=look_up_table_path)
    gsft.generateTable_save_results(tableFormat=tableFormat)
    start = timeit.default_timer()

    stop = timeit.default_timer()
    print('Time taken to save the table:  {0} min'.format((stop - start) / 60))
