from src.functions.imputation_methods import *
import os
import pyopenms as oms

consensus_map = oms.ConsensusMap()
oms.ConsensusXMLFile().load("/home/pixel/Documents/Cinvestav_2025/UI_Analisis_Genomico/Consensus/normalized.consensusXML", consensus_map)

# Extraer metadatos de los archivos originales
column_headers = consensus_map.getColumnHeaders()
sorted_columns = sorted(column_headers.items(), key=lambda x: x[0])

# Modificar esta línea para obtener solo el nombre del archivo
filenames = [os.path.basename(header.filename) for idx, header in sorted_columns] 

# Construir el DataFrame
rows = []
for cf in consensus_map:
    row = {
        'rt': cf.getRT(),
        'mz': cf.getMZ(),
        'intensity': cf.getIntensity()
    }
    
    # Inicializar intensidades con NaN
    for filename in filenames:
        row[filename] = float('nan')
    
    # Llenar intensidades de cada archivo
    for fh in cf.getFeatureList():
        map_idx = fh.getMapIndex()
        if map_idx < len(filenames):
            filename = filenames[map_idx]  # Ahora usa solo el nombre corto
            row[filename] = fh.getIntensity()
    
    rows.append(row)

# Crear DataFrame y ordenar columnas
df = pd.DataFrame(rows)
columns = ['rt', 'mz', 'intensity'] + filenames
df = df[columns]

df = df.drop(columns=['rt', 'mz', 'intensity'])

print("Original DataFrame:")
print(df)
print("\nMissing values before imputation:")
print(df.isnull().sum())

# Impute missing values
#df_imputed = nImputed(df)
#df_imputed = nImputed(df,100)
#df_imputed = halfMinimumImputed(df)
#df_imputed = meanImputed(df)
#df_imputed = medianImputed(df)

#df_imputed = missForestImputed(df)
#df_imputed = missForestImputed(df,1,50)
#df_imputed = svdImputed(df)
#df_imputed = knnImputed(df,2)
#df_imputed = knnImputed(df,5)
#df_imputed = miceLinearRegressionImputed(df)
#df_imputed = miceBayesianRidgeImputed(df)

# Always scale before imputation for advanced methods
df_scaled = (df - df.mean()) / df.std()

#df_imputed = missForestImputed(df_scaled).pipe(postprocess_imputation, df)
#df_imputed = svdImputed(df_scaled).pipe(postprocess_imputation, df)
#df_imputed = knnImputed(df_scaled).pipe(postprocess_imputation, df)
df_imputed = miceLinearRegressionImputed(df_scaled).pipe(postprocess_imputation, df)
#df_imputed = miceBayesianRidgeImputed(df_scaled).pipe(postprocess_imputation, df)

print("\nDataFrame after imputation:")
print(df_imputed)
print("\nMissing values after imputation:")
print(df_imputed.isnull().sum())