import pandas as pd
path = r"c:\Users\oussa\OneDrive\Desktop\Donnees_Nettoyees (6).xlsx"
try:
    xls = pd.ExcelFile(path)
    print('SHEETS:', xls.sheet_names)
    for s in xls.sheet_names:
        try:
            df = pd.read_excel(path, sheet_name=s)
            print(f"SHEET: {s!r} -> rows={len(df)}, cols={len(df.columns)}")
        except Exception as e:
            print(f"SHEET: {s!r} -> ERROR reading: {e}")
except Exception as e:
    print('ERROR', e)
