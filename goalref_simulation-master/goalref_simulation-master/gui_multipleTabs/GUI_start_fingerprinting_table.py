from __future__ import print_function
import timeit
import GUI_config_fingerprinting_table as config_fingerprinting
import GUI_config_MultipleTabs as config_objects
from multiprocessing import freeze_support
from GUI_class_generate_fingerprinting_table import GenerateFingerprintingTable
__author__ = 'ihm'

# This is script does the following:
#--> Create a GernerateFingerprintingTable Object
#--> Passing the configurations from both GUI_config_fingerprinting_table and
# GUI_config_MultipleTabs to that object
#--> Calculate the fingerprinting table using calculate_table() method
#--> Save the calculated table using save_results() method
if __name__ == '__main__':
    freeze_support()
    print('\nApplication: ' + config_fingerprinting.app)
    generateTable = GenerateFingerprintingTable(config_objects.exciter, config_objects.coil,
                                                config_objects.antennaList,
                                                config_objects.frequency, config_objects.coilResistance,
                                                config_fingerprinting.positions, config_fingerprinting.angles,
                                                config_objects.objectType, config_fingerprinting.numberOfCores,
                                                config_fingerprinting.app, config_fingerprinting.author,
                                                config_fingerprinting.additional_info)
    start = timeit.default_timer()
    generateTable.calculate_table()
    stop = timeit.default_timer()
    print('Runtime:  {0} min'.format((stop - start) / 60))

    start = timeit.default_timer()
    generateTable.save_results(path='.\\tables', tableName=config_fingerprinting.tableName,
                               tableFormat=config_fingerprinting.tableFormat)
    stop = timeit.default_timer()
    print('Time taken to save the table:  {0} min'.format((stop - start) / 60))
