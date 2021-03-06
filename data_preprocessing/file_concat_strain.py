import pandas as pd

months = [
    # '0901', '0902', '0903', '0904', '0905', '0906', '0907', '0908', '0909', '0910', '0911', '0912'
    # '1001', '1002', '1003', '1004', '1005', '1006', '1007', '1008', '1009', '1010', '1011', '1012'
    # '1101', '1102', '1103', '1104', '1105', '1106', '1107', '1108', '1109', '1110', '1111', '1112'
    # '1201', '1202', '1203', '1204', '1205', '1206', '1207', '1208', '1209', '1210', '1211', '1212'
    # '1301', '1302', '1303', '1304', '1305', '1306', '1307', '1308', '1309', '1310', '1311', '1312'
    # '1401', '1402', '1403', '1404', '1405', '1406', '1407', '1408', '1409', '1410', '1411', '1412'
    # '1501', '1502', '1503', '1504', '1505', '1506', '1507', '1508', '1509', '1510', '1511', '1512'
    '1601', '1602', '1603', '1604', '1605', '1606', '1607', '1608', '1609', '1610', '1611', '1612'
]
year = '2016'

list_data = []
for month in months:
    print month
    # data = pd.read_csv('../data/new_strain/ht%s.csv' % month, header=None,
    #                    names=['time', 'radial_strain', 'tangential_strain'], index_col=False)
    data = pd.read_csv('../data/new_strain/ht%s.csv' % month)
    time = data[data.columns[0]].values
    radial = data[data.columns[61]].values
    tangential = data[data.columns[62]].values
    new_data = pd.DataFrame({"time": time, "radial_strain": radial, "tangential_strain": tangential})
    list_data.append(new_data)

all_data = pd.concat(list_data)
all_data.to_csv('../data/new_strain/data_%s.csv' % year, index=None)
