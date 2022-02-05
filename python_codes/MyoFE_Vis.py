import json
import pandas as pd
import sys

from PyCMLutil.plots.multi_panel import multi_panel_from_flat_data as mpl


def MyoFE_vis():
    no_of_arguments = len(sys.argv)
    print(no_of_arguments)

    if no_of_arguments == 1:
       print ("No argument is called")

    elif no_of_arguments == 2:
        instruction_file_str = sys.argv[1]
        generate_figures(instruction_file_str)

def generate_figures(instruction_file_str = []):
    if not instruction_file_str == []:
        with open(instruction_file_str,'r') as ins:
            instruction_data = json.load(ins)

    images_handler_list = []

    if 'sim_results_file_string' in instruction_data:
            sim_data = \
                pd.read_csv(instruction_data['sim_results_file_string'])

            print(sim_data.head())

    if 'output_image_handler' in instruction_data:
            for i in instruction_data['output_image_handler']:
                images_handler_list.append(i)

    for image_data in images_handler_list:
            temp_str = image_data['template_instruction_str']
            output_image_str = image_data['output_image_str']
            mpl(pandas_data = sim_data,
                template_file_string=temp_str,
                output_image_file_string = output_image_str)

if __name__=='__main__':
    MyoFE_vis()


        