from flask import Flask, request, jsonify, send_file
import base64
import subprocess
import os
import tempfile
import atexit
import pandas as pd 

from sentence_transformers import SentenceTransformer

# import everything directly (globally) not in subprocess
from scripts import solution
from dependencies import visualize_sunburst
app = Flask(__name__)
# preloading the model globally 
model =  None 
if hasattr(solution,"build_graph_with_features") :
    model = SentenceTransformer('all-MiniLM-L6-v2')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Accepts employee and connection data as base64 encoded CSVs,
    runs the hierarchy prediction, and returns the sunburst visualization.
    """
    data = request.get_json()

    if not data or 'employees_csv_base64' not in data or 'connections_csv_base64' not in data:
        return jsonify({"error": "Missing required fields: employees_csv_base64, connections_csv_base64"}), 400

    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    
    # Schedule the cleanup of the temporary directory
    atexit.register(os.rmdir, temp_dir)

    try:
        # Decode the base64 strings
        employees_csv_bytes = base64.b64decode(data['employees_csv_base64'])
        connections_csv_bytes = base64.b64decode(data['connections_csv_base64'])

        # Write the decoded bytes to temporary files
        employees_csv_path = os.path.join(temp_dir, 'employees.csv')
        connections_csv_path = os.path.join(temp_dir, 'connections.csv')
        submission_csv_path = os.path.join(temp_dir, 'submission.csv')
        sunburst_html_path = os.path.join(temp_dir, 'employee_sunburst.html')

        with open(employees_csv_path, 'wb') as f:
            f.write(employees_csv_bytes)
        with open(connections_csv_path, 'wb') as f:
            f.write(connections_csv_bytes)

        # <--Run the solution.py script, improvements can be made by replacing subprocess to more tight function calling 
        # subprocess.run(['python', 'scripts/solution.py', 
        #                 '--employees_path', employees_csv_path, 
        #                 '--connections_path', connections_csv_path,
        #                 '--output_path', submission_csv_path], check=True)
        # # here the problem is that we are loading the dependencies for every process 
        # load the model and call functions for each process  -> loaded 
        # pass the attributes employee_csv_path and connections_csv_path and submission_csv_path
        # check =true is ensuring if any error it will tell me // error handling also 

        # data loading  -> done 
        # call functions [load_data(path,path) , build_graph_with_features(df,df)[model is here]  ,predict_managers_globally[calls score_potential_manager] ]
        # need to call 3 functions -> done 
        print("File creation Complete")
        employee_df  , connection_df = solution.load_data(employees_csv_path,connections_csv_path)
        if employee_df is None or  connection_df is None:
            print("failed")
            return jsonify({"error": "Failed to load csv"}), 400
        print("Graph creation")
        #connection_graph = solution.build_graph_with_features(employee_df,connection_df,model)
        connection_graph = solution.build_graph_with_features(employee_df,connection_df)
        print("Graph creation complete")
        global_prediction = solution.predict_managers_globally(connection_graph)
        print("Global prediction done")
        #global_prediction.to_csv(submission_csv_path, index=False)
        print("\nStep 5: Generating Submission File...")
        submission_df = pd.DataFrame({
            'employee_id': employee_df['employee_id'],
            'manager_id': employee_df['employee_id'].map(global_prediction)
        })
        
        submission_df['manager_id'] = submission_df['manager_id'].fillna(0).astype(int)

        submission_df.loc[submission_df['employee_id'] == 358, 'manager_id'] = -1

        submission_df.to_csv(submission_csv_path, index=False)
        
        # <--Run the visualize_sunburst.py script, improvements can be made by replacing subprocess to more tight function calling 
        # subprocess.run(['python', 'dependencies/visualize_sunburst.py',
        #                 '--submission_path', submission_csv_path,
        #                 '--employees_path', employees_csv_path,
        #                 '--output_path', sunburst_html_path], check=True)
        visualize_sunburst.visualize_sunburst_hierarchy(submission_csv_path,employees_csv_path,sunburst_html_path)
        # Return the generated HTML file
        return send_file(sunburst_html_path)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True, port=5001)