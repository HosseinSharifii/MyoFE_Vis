from email.mime import base
import json
import os
import numpy as np
import pandas as pd
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from PyCMLutil.plots.multi_panel import multi_panel_from_flat_data as mpl
from PyCMLutil.plots.pv_plots import display_pv_loop as pv

def return_spatial_data_path():

    data = []
    data.append("../working_dir/base_sim/sim_output/p_mmhg/test_20/full/cb_stress_data.csv")
    data.append("../working_dir/base_sim/sim_output/p_mmhg/test_20/full/hs_length_data.csv")
    return data
def MyoFE_vis():
    no_of_arguments = len(sys.argv)
    print(no_of_arguments)

    if no_of_arguments == 1:
       print ("No argument is called")

    elif no_of_arguments == 2:
        if sys.argv[1] == 'pv':
            test_pv()
        else:
            instruction_file_str = sys.argv[1]
            generate_figures(instruction_file_str)
            
    
    elif no_of_arguments == 3:
        if sys.argv[1] == 'spatial':
            output_figure_str = sys.argv[2]
            generate_spatial_figures(output_figure_str)

def generate_figures(instruction_file_str = []):
    instruction_data = []
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

def generate_spatial_figures(output_figure_str = "",dpi = 300):
    
    data = return_spatial_data_path()

    fig = plt.figure(constrained_layout = True)
    fig.set_size_inches(6,6)
    spec = gridspec.GridSpec(nrows = len(data),
                            ncols = 1,
                            figure = fig,
                            wspace = 0.1,
                            hspace = 0.1)
    y = np.array(range(5228))
    print (y)
    for i, d in enumerate(data):
        data_frame = pd.read_csv(d)
        cols = np.append(y,['time'])
        data_frame = data_frame[cols]
        
       
        y_keys = list(map(str,y))

        data_frame['avg'] = data_frame[y_keys].mean(axis = 1)
        data_frame['std_pos'] = data_frame['avg']+data_frame[y_keys].std(axis=1)
        data_frame['std_neg'] = data_frame['avg']-data_frame[y_keys].std(axis=1)
        print(data_frame.head())
        ax = fig.add_subplot(spec[i,0])
        x = data_frame['time']
        y_data = data_frame['avg']
        ax.plot(x, y_data)
        ax.fill_between(x,data_frame['std_pos'],data_frame['std_neg'],alpha = 0.5)
        #sns.lineplot(ax= ax,data = data_frame,x='time',y=y_keys)

    if output_figure_str:
        output_dir = os.path.dirname(output_figure_str)
        if not os.path.isdir(output_dir):
            print('Making output dir')
            os.makedirs(output_dir)
        fig.savefig(output_figure_str,dpi=dpi)
        
    return

def test_pv():
    data_str = ['../working_dir/base_sim/sim_output/p_mmhg/test_18/data.csv']
    time_frames = [(5,6)]
    out_put = '../working_dir/base_sim/sim_output/p_mmhg/test_18/pv_test.png'
    pv(data_file_string = data_str,
        time_frames = time_frames,
        pressure_var_name = 'pressure_ventricle',
        volume_var_name = 'volume_ventricle',
        template_data = {'formatting':{'palette':None,'legend_bbox_to_anchor':[0.7,1]},
                        'layout':{'fig_width':6}},
        time_var_name = 'time',
        legend_labels = ['PV loop'],
        y_label = 'Pressure\n(mmHg)',
        output_image_file_string = out_put,
        dpi = 300)
def check_pv_loop():
    data_str = ['../working_dir/base_sim/sim_output/mpi_testing/new_test/data.csv',
                '../working_dir/base_sim/sim_output/mpi_testing/baro_set_down/data.csv',
                '../working_dir/base_sim/sim_output/mpi_testing/AS_nice/data.csv']
    data = pd.DataFrame()

    for i,st in enumerate(data_str):
        t = [4.5,5.5]
        temp_data = pd.read_csv(st)
        temp_data = temp_data[temp_data['time'].between(t[0],t[1])]
        data['pressure_'+str(i+1)] = temp_data['pressure_ventricle']
        data['volume_'+str(i+1)] = temp_data['volume_ventricle']

    data['time'] = temp_data['time']
    print(data)

    temp_str = '../working_dir/base_sim/sim_inputs/temp/template_compare_pv.json'
    output_pv_str = '../working_dir/base_sim/sim_output/compare_pv.png'
    mpl(pandas_data = data,
                template_file_string=temp_str,
                output_image_file_string = output_pv_str)
    
    data['delta_p_12'] = data['pressure_1'] - data['pressure_2']
    data['delta_p_13'] = data['pressure_1'] - data['pressure_3']
    data['delta_p_23'] = data['pressure_2'] - data['pressure_3']

    data['delta_v_12'] = data['volume_1'] - data['volume_2']
    data['delta_v_13'] = data['volume_1'] - data['volume_3']
    data['delta_v_23'] = data['volume_2'] - data['volume_3']
    
    temp_str = '../working_dir/base_sim/sim_inputs/temp/template_pv_over_time.json'
    output_pv_time_str = '../working_dir/base_sim/sim_output/pv_time.png'
    mpl(pandas_data = data,
                template_file_string=temp_str,
                output_image_file_string = output_pv_time_str)
    
def extract_pv_loop_data(folder_str = ""):
    t = [3.5,4.5]

    tempelate_str = '../working_dir/base_sim/sim_inputs/temp/template_compare_pv.json'
    with open(tempelate_str,'r') as t:
        base_template_file = json.load(t)
    base_template_file['panels'][0]['y_info']['series'] = []
    base_template_file['x_display']['global_x_field'] = 'volume_1_cores_1'
    data = pd.DataFrame()
    for f in os.listdir(folder_str):
        temp_str =folder_str + f 
        # check if output exist
       
        if not f == '.DS_Store' and 'sim_output' in os.listdir(temp_str):
            temp_data_str = temp_str + '/sim_output/data.csv'
            
            temp_data = pd.read_csv(temp_data_str)
            temp_data = temp_data[temp_data['time'].between(t[0],t[1])]
            data['pressure_'+f] = temp_data['pressure_ventricle']
            data['volume_'+f] = temp_data['volume_ventricle']

            # now complete template file
            temp_dict = dict()
            temp_dict['field'] = 'pressure_'+f
            temp_dict['style'] = 'line'
            temp_dict['style'] = f
            base_template_file['panels'][0]['y_info']['series'].append(temp_dict)
    
    data['time'] = temp_data['time']
    print(data)
    output_data = '../working_dir/base_sim/sim_output/mpi_sensitivity/extracted_pv_data.csv'
    data.to_csv(output_data)

    new_template_str = '../working_dir/base_sim/sim_output/mpi_sensitivity/demos/template.json'
    with open(new_template_str,'w') as to:
        json.dump(base_template_file,to,indent=4)
    
        
    
    return output_data, new_template_str

def generate_template_file(folder_str = "",
                           template_file_str = ""):
    
    
    with open(template_file_str,'r') as t:
        base_template_file = json.load(t)
    base_template_file['panels'][0]['y_info']['series'] = []
    base_template_file['x_display']['global_x_field'] = 'volume_1_cores_5'
    base_template_file['formatting']['max_rows_per_legend'] = 25
    base_template_file['formatting']['legend_bbox_to_anchor'] = [0.65, 1.05]
    base_template_file['formatting']['legend_fontsize'] = 7
    for f in os.listdir(folder_str):
        temp_str =folder_str + f 
        # check if output exist
        if not f == '.DS_Store' and 'sim_output' in os.listdir(temp_str):
            # now complete template file
            temp_dict = dict()
            temp_dict['field'] = 'pressure_'+f
            temp_dict['style'] = 'line'
            temp_dict['field_label'] = f
            temp_dict['field_color'] = 'black'
            base_template_file['panels'][0]['y_info']['series'].append(temp_dict)
    new_template_str = '../working_dir/base_sim/sim_output/mpi_sensitivity/template.json'
    with open(new_template_str,'w') as to:
        json.dump(base_template_file,to,indent=4)

    return

def compute_avg_and_std(data_str = '',
                        output_data = ''):
    df = pd.read_csv(data_str)
    pressure_columns = []
    volume_columns = []
    p_32_cores_cols = []
    p_16_cores_cols = []
    p_8_cores_cols = []
    p_4_cores_cols = []
    p_2_cores_cols = []
    v_32_cores_cols = []
    v_16_cores_cols = []
    v_8_cores_cols = []
    v_4_cores_cols = []
    v_2_cores_cols = []
    for col in df.columns:
        if col.split('_')[0] == 'pressure':
            pressure_columns.append(col)
            if col.split('pressure_')[1].split('_')[0] == '32':
                p_32_cores_cols.append(col)
            if col.split('pressure_')[1].split('_')[0] == '16':
                p_16_cores_cols.append(col)
            if col.split('pressure_')[1].split('_')[0] == '8':
                p_8_cores_cols.append(col)
            if col.split('pressure_')[1].split('_')[0] == '4':
                p_4_cores_cols.append(col)
            if col.split('pressure_')[1].split('_')[0] == '2':
                p_2_cores_cols.append(col)
        if col.split('_')[0] == 'volume':
            volume_columns.append(col)
            if col.split('volume_')[1].split('_')[0] == '32':
                v_32_cores_cols.append(col)
            if col.split('volume_')[1].split('_')[0] == '16':
                v_16_cores_cols.append(col)
            if col.split('volume_')[1].split('_')[0] == '8':
                v_8_cores_cols.append(col)
            if col.split('volume_')[1].split('_')[0] == '4':
                v_4_cores_cols.append(col)
            if col.split('volume_')[1].split('_')[0] == '2':
                v_2_cores_cols.append(col)
    print(p_32_cores_cols)
    df['pressure_avg'] = df[pressure_columns].mean(axis=1)
    df['pressure_std'] = df[pressure_columns].std(axis=1)
    df['pressure_pos_std'] = df['pressure_avg'] + df['pressure_std']
    df['pressure_neg_std'] = df['pressure_avg'] - df['pressure_std']
    df['volume_avg'] = df[volume_columns].mean(axis=1)
    df['volume_std'] = df[volume_columns].std(axis=1)
    df['volume_pos_std'] = df['volume_avg'] + df['volume_std']
    df['volume_neg_std'] = df['volume_avg'] - df['volume_std']

    df['p_avg_32_cores'] = df[p_32_cores_cols].mean(axis=1)
    df['p_avg_16_cores'] = df[p_16_cores_cols].mean(axis=1)
    df['p_avg_8_cores'] = df[p_8_cores_cols].mean(axis=1)
    df['p_avg_4_cores'] = df[p_4_cores_cols].mean(axis=1)
    df['p_avg_2_cores'] = df[p_2_cores_cols].mean(axis=1)

    df['v_avg_32_cores'] = df[v_32_cores_cols].mean(axis=1)
    df['v_avg_16_cores'] = df[v_16_cores_cols].mean(axis=1)
    df['v_avg_8_cores'] = df[v_8_cores_cols].mean(axis=1)
    df['v_avg_4_cores'] = df[v_4_cores_cols].mean(axis=1)
    df['v_avg_2_cores'] = df[v_2_cores_cols].mean(axis=1)

    df['dif_p_32_1'] = (df['p_avg_32_cores'] - df['pressure_1_cores_no_mpi_1']).abs()/df['pressure_1_cores_no_mpi_1']
    df['dif_p_16_1'] = (df['p_avg_16_cores'] - df['pressure_1_cores_no_mpi_1']).abs()/df['pressure_1_cores_no_mpi_1']
    df['dif_p_8_1'] = (df['p_avg_8_cores'] - df['pressure_1_cores_no_mpi_1']).abs()/df['pressure_1_cores_no_mpi_1']
    df['dif_p_4_1'] = (df['p_avg_4_cores'] - df['pressure_1_cores_no_mpi_1']).abs()/df['pressure_1_cores_no_mpi_1']
    df['dif_p_2_1'] = (df['p_avg_2_cores'] - df['pressure_1_cores_no_mpi_1']).abs()/df['pressure_1_cores_no_mpi_1']
    df['dif_p_1_1'] = (df['pressure_1_cores_no_mpi_2'] - df['pressure_1_cores_no_mpi_1']).abs()/df['pressure_1_cores_no_mpi_1']

    df['dif_v_32_1'] = (df['v_avg_32_cores'] - df['volume_1_cores_no_mpi_1']).abs()/df['volume_1_cores_no_mpi_1']
    df['dif_v_16_1'] = (df['v_avg_16_cores'] - df['volume_1_cores_no_mpi_1']).abs()/df['volume_1_cores_no_mpi_1']
    df['dif_v_8_1'] = (df['v_avg_8_cores'] - df['volume_1_cores_no_mpi_1']).abs()/df['volume_1_cores_no_mpi_1']
    df['dif_v_4_1'] = (df['v_avg_4_cores'] - df['volume_1_cores_no_mpi_1']).abs()/df['volume_1_cores_no_mpi_1']
    df['dif_v_2_1'] = (df['v_avg_2_cores'] - df['volume_1_cores_no_mpi_1']).abs()/df['volume_1_cores_no_mpi_1']
    df['dif_v_1_1'] = (df['volume_1_cores_no_mpi_2'] - df['volume_1_cores_no_mpi_1']).abs()/df['volume_1_cores_no_mpi_1']

    """df['dif_p_32_1'] = (df['pressure_32_cores_1'] - df['pressure_1_cores_no_mpi_1']).abs()/df['pressure_1_cores_no_mpi_1']
    df['dif_p_16_1'] = (df['pressure_16_cores_1'] - df['pressure_1_cores_no_mpi_1']).abs()/df['pressure_1_cores_no_mpi_1']
    df['dif_p_8_1'] = (df['pressure_8_cores_1'] - df['pressure_1_cores_no_mpi_1']).abs()/df['pressure_1_cores_no_mpi_1']
    df['dif_p_4_1'] = (df['pressure_4_cores_1'] - df['pressure_1_cores_no_mpi_1']).abs()/df['pressure_1_cores_no_mpi_1']
    df['dif_p_2_1'] = (df['pressure_2_cores_1'] - df['pressure_1_cores_no_mpi_1']).abs()/df['pressure_1_cores_no_mpi_1']

    df['dif_v_32_1'] = (df['volume_32_cores_1'] - df['volume_1_cores_no_mpi_1']).abs()/df['volume_1_cores_no_mpi_1']
    df['dif_v_16_1'] = (df['volume_16_cores_1'] - df['volume_1_cores_no_mpi_1']).abs()/df['volume_1_cores_no_mpi_1']
    df['dif_v_8_1'] = (df['volume_8_cores_1'] - df['volume_1_cores_no_mpi_1']).abs()/df['volume_1_cores_no_mpi_1']
    df['dif_v_4_1'] = (df['volume_4_cores_1'] - df['volume_1_cores_no_mpi_1']).abs()/df['volume_1_cores_no_mpi_1']
    df['dif_v_2_1'] = (df['volume_2_cores_1'] - df['volume_1_cores_no_mpi_1']).abs()/df['volume_1_cores_no_mpi_1']"""


    print(df['pressure_avg'])
    df.to_csv(output_data)
    
    return

if __name__=='__main__':
    #MyoFE_vis()
    #check_pv_loop()
    folder_str = \
        '../working_dir/base_sim/sim_output/mpi_sensitivity/demos/'
    #extract_pv_loop_data(folder_str = folder_str)

    tempelate_str = '../working_dir/base_sim/sim_inputs/temp/template_compare_pv.json'

    generate_template_file(folder_str = folder_str,template_file_str = tempelate_str)

    data_str = '../working_dir/base_sim/sim_output/mpi_sensitivity/extracted_pv_data.csv'
    pandas_data = pd.read_csv(data_str)
    temp_str = '../working_dir/base_sim/sim_output/mpi_sensitivity/template.json'
    output_pv_time_str = '../working_dir/base_sim/sim_output/mpi_sensitivity/pv_mpi.png'
    mpl(pandas_data = pandas_data,
                template_file_string=temp_str,
                output_image_file_string = output_pv_time_str)
    
    output_data = '../working_dir/base_sim/sim_output/mpi_sensitivity/extracted_data_with_avg_std.csv'
    compute_avg_and_std(data_str = data_str,
                        output_data = output_data)

    avg_data = pd.read_csv(output_data)
    temp_str = '../working_dir/base_sim/sim_output/mpi_sensitivity/template_std.json'
    output_std = '../working_dir/base_sim/sim_output/mpi_sensitivity/std.png'
    mpl(pandas_data = avg_data,
                template_file_string=temp_str,
                output_image_file_string = output_std)
    
    temp_str = '../working_dir/base_sim/sim_output/mpi_sensitivity/template_diff.json'
    output_dif = '../working_dir/base_sim/sim_output/mpi_sensitivity/dif.png'
    mpl(pandas_data = avg_data,
                template_file_string=temp_str,
                output_image_file_string = output_dif)
    
    temp_str = '../working_dir/base_sim/sim_output/mpi_sensitivity/template_P_time.json'
    output_dif = '../working_dir/base_sim/sim_output/mpi_sensitivity/1core_vs_2cores.png'
    mpl(pandas_data = avg_data,
                template_file_string=temp_str,
                output_image_file_string = output_dif)

    temp_str = '../working_dir/base_sim/sim_output/mpi_sensitivity/template_P_time_1.json'
    output_dif = '../working_dir/base_sim/sim_output/mpi_sensitivity/1core_vs_1core.png'
    mpl(pandas_data = avg_data,
                template_file_string=temp_str,
                output_image_file_string = output_dif)