import pandas as pd
from flask_cors import CORS, cross_origin
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from flask import Flask, request, render_template
import pickle

#Create an app object using the Flask class. 
app = Flask(__name__)


#Load the trained model. (Pickle file)
model = pickle.load(open('model.pkl', 'rb'))
@app.route('/')
def home():
    return render_template('index.html')


# @app.route('/predict', methods=['POST'])
# def predict():
#     file = request.files['cpufile1']
#     if file:
#         df = pd.read_csv(file)  # Use only the first 58 tuples
#         y_test = df.iloc[-1, :].values  # Use only the first 58 tuples

#         X = []
#         for col in df.columns:
#             data = df[col].to_numpy()
#             X.append(data[:287])
#         X_test = np.array(X)

#         # Add necessary reshaping for LSTM input

#         X_test_reshaped = X_test.reshape(X_test.shape[0],
#                 X_test.shape[1], 1)

#         # Predict the target variable using the trained model

#         y_pred = model.predict(X_test_reshaped).flatten()
#         n = len(y_pred)
        
#         mse = mean_squared_error(y_test, y_pred)/n
#         mad = mean_absolute_error(y_test, y_pred)/n
#         r2 = r2_score(y_test, y_pred)

#         y_pred_split = np.array_split(y_pred, 6)
     
#         # Initialize lists to store server status and predictions for each split
#         total_sum_predicted = sum(sum(split) for split in y_pred_split)

#         server_status_list = []
#         predictions_list = []

#         for (i, split) in enumerate(y_pred_split):
#             sum_predicted_values_split = np.sum(split)
#             split_length = len(split)
#             underloaded_threshold = split_length * 20
#             overloaded_threshold = split_length * 80 
#             server_status = ''
#             if sum_predicted_values_split < underloaded_threshold:
#                 server_status = ' is going to be underloaded'
#             elif sum_predicted_values_split > overloaded_threshold:
#                 server_status = ' is going to be overloaded'
#             else:
#                 server_status = ' is going to run normally'
                
                
#             # Append the server status to the list

#             server_status_list.append(server_status)

#             # Append the predictions for the current split

#             predictions_list.append(split.tolist())
            
    
#     return render_template(
#         'try2.html',
#         predictions=predictions_list,
#         mse=mse,
#         mad=mad,
#         r2=r2,
#         server_status=server_status_list,
#         )


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['cpufile1']
    if file:
        df = pd.read_csv(file)  # Use only the first 58 tuples
        y_test = df.iloc[-1, :].values  # Use only the first 58 tuples

        X = []
        for col in df.columns:
            data = df[col].to_numpy()
            X.append(data[:287])
        X_test = np.array(X)

        # Add necessary reshaping for LSTM input
        X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        # Predict the target variable using the trained model
        y_pred = model.predict(X_test_reshaped).flatten()
        n = len(y_pred)
        
        mse = mean_squared_error(y_test, y_pred)/len(y_test)
        mad = mean_absolute_error(y_test, y_pred)/len(y_test)
        r2 = r2_score(y_test, y_pred)

        y_pred_split = np.array_split(y_pred, 6)
     
        # Initialize lists to store server status and predictions for each split
        total_sum_predicted = sum(sum(split) for split in y_pred_split)

        server_status_list = []
        predictions_list = []

        for (i, split) in enumerate(y_pred_split):
            sum_predicted_values_split = np.sum(split)
            split_length = len(split)
            underloaded_threshold = split_length * 20
            overloaded_threshold = split_length * 80 
            server_status = ''
            if sum_predicted_values_split < underloaded_threshold:
                server_status = ' is going to be underloaded'
            elif sum_predicted_values_split > overloaded_threshold:
                server_status = ' is going to be overloaded'
            else:
                server_status = ' is going to run normally'
                
            # Append the server status to the list
            server_status_list.append(server_status)

            # Append the predictions for the current split
            predictions_list.append(split.tolist())

        # VM migration logic
                # VM migration logic
        # Define server capacities
        server_capacities = [40 + np.random.randint(61) for _ in range(6)]  # Assuming 6 servers

        # Identify underloaded and overloaded servers
        underloaded_servers = [i for i, split in enumerate(y_pred_split) if np.sum(split) < (len(split) * 20)]
        overloaded_servers = [i for i, split in enumerate(y_pred_split) if np.sum(split) > (len(split) * 80)]
        
        # available_servers.sort(key=lambda x: server_capacities[x], reverse=True)

        # Perform VM migration
        # Perform VM migration
        # migration_info = []

        # # Migrate overloaded servers
        # for overloaded_server in overloaded_servers:
        #     for idx, available_capacity in enumerate(server_capacities):
        #         if available_capacity > 0:
        #             # Calculate the number of VMs that can be migrated
        #             num_migrate_vms = min(np.sum(y_pred_split[overloaded_server]), available_capacity)

        #             print(f"Index: {idx}, Overloaded Server: {overloaded_server}, Num Migrate VMs: {num_migrate_vms}")

        #             if num_migrate_vms > 0:
        #                 # Pad the arrays to ensure they have the same length
        #                 pad_width = max(0, int(len(y_pred_split[underloaded_server][:int(num_migrate_vms)]) - len(y_pred_split[idx])))
        #                 padded_array = np.pad(y_pred_split[idx], (0, pad_width), mode='constant')

        #                 # Migrate VMs from overloaded server to available server
        #                 y_pred_split[idx] += padded_array
        #                 y_pred_split[overloaded_server][:int(num_migrate_vms)] = 0  # Reset migrated VMs in overloaded server

        #                 # Update server status
        #                 # server_status_list[overloaded_server] = f"Migrated to Server {idx + 1}"
        #                 # server_status_list[idx] = ' is going to run normally'  # Assuming no longer underloaded

        #                 # Record migration information
        #                 migration_info.append(f"VMs migrated from Server {overloaded_server + 1} to Server {idx + 1} ({int(num_migrate_vms)} VMs)")

        #                 # Update server capacity
        #                 server_capacities[idx] -= int(num_migrate_vms)

        #                 # Break inner loop to avoid migrating to multiple available servers
        #                 break

        # # Migrate underloaded servers
        # for underloaded_server in underloaded_servers:
        #     for idx, available_capacity in enumerate(server_capacities):
        #         if available_capacity > 0:
        #             # Calculate the number of VMs that can be migrated
        #             num_migrate_vms = min(np.sum(y_pred_split[underloaded_server]), available_capacity)

        #             print(f"Index: {idx}, Underloaded Server: {underloaded_server}, Num Migrate VMs: {num_migrate_vms}")

        #             if num_migrate_vms > 0:
        #                 # Pad the arrays to ensure they have the same length
        #                 pad_width = max(0, int(len(y_pred_split[underloaded_server][:int(num_migrate_vms)]) - len(y_pred_split[idx])))
        #                 padded_array = np.pad(y_pred_split[idx], (0, pad_width), mode='constant')

        #                 # Migrate VMs from underloaded server to available server
        #                 y_pred_split[idx] += padded_array
        #                 y_pred_split[underloaded_server][:int(num_migrate_vms)] = 0  # Reset migrated VMs in underloaded server

        #                 # Update server status
        #                 server_status_list[underloaded_server] = f"Migrated to Server {idx + 1}"
        #                 server_status_list[idx] = ' is going to run normally'  # Assuming no longer underloaded

        #                 # Record migration information
        #                 migration_info.append(f"VMs migrated from Server {underloaded_server + 1} to Server {idx + 1} ({int(num_migrate_vms)} VMs)")

        #                 # Update server capacity
        #                 server_capacities[idx] -= int(num_migrate_vms)

        #                 # Break inner loop to avoid migrating to multiple available servers
        #                 break

        # # Recalculate total sum of predicted values after migration
        # total_sum_predicted = sum(sum(split) for split in y_pred_split)




        # # Recalculate total sum of predicted values after migratio

        # # Save migration information to CSV file
        
        # migration_df = pd.DataFrame({'Migration Information': migration_info})
        #         # Append mode is used to add data to the CSV file instead of overwriting it
        # with open('migration_info.csv', 'a') as f:
        #     migration_df.to_csv(f, header=False, index=False)

    return render_template(
        'try2.html',
        predictions=predictions_list,
        mse=mse,
        mad=mad,
        r2=r2,
        server_status=server_status_list,
        total_sum_predicted=total_sum_predicted,
        migration_info=migration_info
    )

@app.route('/migration_info', methods=['POST'])
def migration_info():
    file = request.files['cpufile1']
    if file:
        df = pd.read_csv(file)  # Use only the first 58 tuples
        y_test = df.iloc[-1, :].values  # Use only the first 58 tuples

        X = []
        for col in df.columns:
            data = df[col].to_numpy()
            X.append(data[:287])
        X_test = np.array(X)

        # Add necessary reshaping for LSTM input
        X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        # Predict the target variable using the trained model
        y_pred = model.predict(X_test_reshaped).flatten()
        n = len(y_pred)
        
        mse = mean_squared_error(y_test, y_pred)/n
        mad = mean_absolute_error(y_test, y_pred)/n
        r2 = r2_score(y_test, y_pred)

        y_pred_split = np.array_split(y_pred, 6)
     
        # Initialize lists to store server status and predictions for each split
        total_sum_predicted = sum(sum(split) for split in y_pred_split)

        server_status_list = []
        predictions_list = []

        for (i, split) in enumerate(y_pred_split):
            sum_predicted_values_split = np.sum(split)
            split_length = len(split)
            underloaded_threshold = split_length * 20
            overloaded_threshold = split_length * 80 
            server_status = ''
            if sum_predicted_values_split < underloaded_threshold:
                server_status = ' is going to be underloaded'
            elif sum_predicted_values_split > overloaded_threshold:
                server_status = ' is going to be overloaded'
            else:
                server_status = ' is going to run normally'
                
            # Append the server status to the list
            server_status_list.append(server_status)

            # Append the predictions for the current split
            predictions_list.append(split.tolist())

        # VM migration logic
                # VM migration logic
        # Define server capacities
        server_capacities = [400 + np.random.randint(61) for _ in range(6)]  # Assuming 6 servers

        # Identify underloaded and overloaded servers
        underloaded_servers = [i for i, split in enumerate(y_pred_split) if np.sum(split) < (len(split) * 20)]
        overloaded_servers = [i for i, split in enumerate(y_pred_split) if np.sum(split) > (len(split) * 80)]
        
        # available_servers.sort(key=lambda x: server_capacities[x], reverse=True)

        migration_info = []

        # Migrate overloaded servers
        for overloaded_server in overloaded_servers:
            for idx, available_capacity in enumerate(server_capacities):
                if available_capacity > 0:
                    # Calculate the number of VMs that can be migrated
                    num_migrate_vms = min(np.sum(y_pred_split[overloaded_server]), available_capacity)

                    print(f"Index: {idx}, Overloaded Server: {overloaded_server}, Num Migrate VMs: {num_migrate_vms}")

                    if num_migrate_vms > 0:
                        # Pad the arrays to ensure they have the same length
                        pad_width = max(0, int(len(y_pred_split[underloaded_server][:int(num_migrate_vms)]) - len(y_pred_split[idx])))
                        padded_array = np.pad(y_pred_split[idx], (0, pad_width), mode='constant')

                        # Migrate VMs from overloaded server to available server
                        y_pred_split[idx] += padded_array
                        y_pred_split[overloaded_server][:int(num_migrate_vms)] = 0  # Reset migrated VMs in overloaded server

                        # Update server status
                        # server_status_list[overloaded_server] = f"Migrated to Server {idx + 1}"
                        # server_status_list[idx] = ' is going to run normally'  # Assuming no longer underloaded

                        # Record migration information
                        migration_info.append(f"VMs migrated from Server {overloaded_server + 1} to Server {idx + 1} ({int(num_migrate_vms)} VMs)")

                        # Update server capacity
                        server_capacities[idx] -= int(num_migrate_vms)

                        # Break inner loop to avoid migrating to multiple available servers
                        break

        # Migrate underloaded servers
        for underloaded_server in underloaded_servers:
            for idx, available_capacity in enumerate(server_capacities):
                if available_capacity > 0:
                    # Calculate the number of VMs that can be migrated
                    num_migrate_vms = min(np.sum(y_pred_split[underloaded_server]), available_capacity)

                    print(f"Index: {idx}, Underloaded Server: {underloaded_server}, Num Migrate VMs: {num_migrate_vms}")

                    if num_migrate_vms > 0:
                        # Pad the arrays to ensure they have the same length
                        pad_width = max(0, int(len(y_pred_split[underloaded_server][:int(num_migrate_vms)]) - len(y_pred_split[idx])))
                        padded_array = np.pad(y_pred_split[idx], (0, pad_width), mode='constant')

                        # Migrate VMs from underloaded server to available server
                        y_pred_split[idx] += padded_array
                        y_pred_split[underloaded_server][:int(num_migrate_vms)] = 0  # Reset migrated VMs in underloaded server

                        # Update server status
                        server_status_list[underloaded_server] = f"Migrated to Server {idx + 1}"
                        server_status_list[idx] = ' is going to run normally'  # Assuming no longer underloaded

                        # Record migration information
                        migration_info.append(f"VMs migrated from Server {underloaded_server + 1} to Server {idx + 1} ({int(num_migrate_vms)} VMs)")

                        # Update server capacity
                        server_capacities[idx] -= int(num_migrate_vms)

                        # Break inner loop to avoid migrating to multiple available servers
                        break

        # Recalculate total sum of predicted values after migration
        total_sum_predicted = sum(sum(split) for split in y_pred_split)



        # Recalculate total sum of predicted values after migratio

        # Save migration information to CSV file
        
        migration_df = pd.DataFrame({'Migration Information': migration_info})
                # Append mode is used to add data to the CSV file instead of overwriting it
        with open('migration_info.csv', 'a') as f:
            migration_df.to_csv(f, header=False, index=False)

    return render_template(
        'migration_info.html',
        predictions=predictions_list,
        mse=mse,
        mad=mad,
        r2=r2,
        server_status=server_status_list,
        total_sum_predicted=total_sum_predicted,
        migration_info=migration_info
    )
    




@app.route('/migration_info2', methods=['POST'])
def migration_info2():
    file = request.files['cpufile1']
    if file:
        df = pd.read_csv(file)  # Use only the first 58 tuples
        y_test = df.iloc[-1, :].values  # Use only the first 58 tuples

        X = []
        for col in df.columns:
            data = df[col].to_numpy()
            X.append(data[:287])
        X_test = np.array(X)

        # Add necessary reshaping for LSTM input
        X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        # Predict the target variable using the trained model
        y_pred = model.predict(X_test_reshaped).flatten()
        n = len(y_pred)
        
        mse = mean_squared_error(y_test, y_pred)/n
        mad = mean_absolute_error(y_test, y_pred)/n
        r2 = r2_score(y_test, y_pred)

        y_pred_split = np.array_split(y_pred, 6)
     
        # Initialize lists to store server status and predictions for each split
        total_sum_predicted = sum(sum(split) for split in y_pred_split)

        server_status_list = []
        predictions_list = []

        for (i, split) in enumerate(y_pred_split):
            sum_predicted_values_split = np.sum(split)
            split_length = len(split)
            underloaded_threshold = split_length * 20
            overloaded_threshold = split_length * 80 
            server_status = ''
            if sum_predicted_values_split < underloaded_threshold:
                server_status = ' is going to be underloaded'
            elif sum_predicted_values_split > overloaded_threshold:
                server_status = ' is going to be overloaded'
            else:
                server_status = ' is going to run normally'
                
            # Append the server status to the list
            server_status_list.append(server_status)

            # Append the predictions for the current split
            predictions_list.append(split.tolist())

        # VM migration logic
                # VM migration logic
        # Define server capacities
        server_capacities = [0.75*n*40 + np.random.randint(61) for _ in range(6)]  # Assuming 6 servers

        # Identify underloaded and overloaded servers
        underloaded_servers = [i for i, split in enumerate(y_pred_split) if np.sum(split) < (len(split) * 20)]
        overloaded_servers = [i for i, split in enumerate(y_pred_split) if np.sum(split) > (len(split) * 80)]
        normal_servers = [i for i, split in enumerate(y_pred_split) if np.sum(split) < (len(split) * 80) and np.sum(split) > (len(split) * 20)]
        # available_servers.sort(key=lambda x: server_capacities[x], reverse=True)

        
# Initialize migration counts
        migration_info = []
        
# Initialize migration counts
        migration_count = 0

        # Migrate VMs from underloaded servers to available servers
        for underloaded_server in underloaded_servers:
            for idx, available_capacity in enumerate(server_capacities):
                if available_capacity > 0:
                    x = len(y_pred_split[underloaded_server])
                    num_migrate_vms = min(np.sum(y_pred_split[underloaded_server]), available_capacity)
                    if num_migrate_vms > 0:
                        server_capacities[idx] -= num_migrate_vms
                        migration_count += x
                        print(f"VMs migrated from Server {underloaded_server+1} to Server {idx+1} ({x} VMs)")
                        migration_info.append(f"VMs migrated from Server {underloaded_server + 1} to Server {idx + 1} ({x} VMs)")
                        break

        # Migrate VMs from overloaded servers to available servers
        for overloaded_server in overloaded_servers:
            for idx, available_capacity in enumerate(server_capacities):
                if available_capacity > 0:
                    y = len(y_pred_split[overloaded_server])
                    num_migrate_vms = min(np.sum(y_pred_split[overloaded_server]), available_capacity)
                    if num_migrate_vms > 0:
                        server_capacities[idx] -= num_migrate_vms
                        migration_count += y
                        print(f"VMs migrated from Server {overloaded_server+1} to Server {idx+1} ({y} VMs)")
                        migration_info.append(f"VMs migrated from Server {overloaded_server + 1} to Server {idx + 1} ({y} VMs)")
                        break
        for server in underloaded_servers:
            print(f"Server {server+1} is underloaded")
        # Shut down servers from which VMs were migrated
        for server in (underloaded_servers + overloaded_servers):
            if server not in normal_servers and server+1 != idx+1:
                print(f"Server {server+1} shuts down")

        # Print total migration count
        print(f"Total VMs migrated: {migration_count}")
        
        shut_down_info = []
        for server in (underloaded_servers + overloaded_servers):
            if server not in normal_servers and server+1 != idx+1:
                print(f"Server {server+1} shuts down")
                shut_down_info.append(f"Server {server+1} shuts down")

        # Recalculate total sum of predicted values after migratio

        # Save migration information to CSV file
        
        migration_df = pd.DataFrame({'Migration Information': migration_info})
                # Append mode is used to add data to the CSV file instead of overwriting it
        with open('migration_info.csv', 'a') as f:
            migration_df.to_csv(f, header=False, index=False)

    return render_template(
        'migration_info2.html',
        predictions=predictions_list,
        mse=mse,
        mad=mad,
        r2=r2,
        server_status=server_status_list,
        total_sum_predicted=total_sum_predicted,
        migration_info=migration_info,
        migration_count=migration_count,
        shut_down_info = shut_down_info
    )
    


@app.route('/other_page')
def other_page():
    return render_template('service.html')

@app.route('/home_page')
def home_page():
    return render_template('index.html')

@app.route('/ffd_page')
def ffd_page():
    return render_template('ffd.html')


@app.route('/bfd_page')
def bfd_page():
    return render_template('bfd.html')


@app.route('/grvmp_page')
def grvmp_page():
    return render_template('grvmp.html')


@app.route('/mbfd_page')
def mbfd_page():
    return render_template('mbfd.html')


@app.route('/newalgo_page')
def newalgo_page():
    return render_template('newalgo.html')


@app.route('/maxmin_page')
def maxmin_page():
    return render_template('maxmin.html')

    
@app.route('/rff_page')
def rff_page():
    return render_template('randomfit.html')


@app.route('/rw_page')
def rw_page():
    return render_template('Roulettewheel.html')


@app.route('/optimize')
def optimize():
    return render_template('optimize.html')


if __name__ == '__main__':
    app.run(port=5000, debug=True)