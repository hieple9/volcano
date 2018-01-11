import pandas as pd

data_type = 'energy'
months = [
     '0901', '0902', '0903', '0904', '0905', '0906', '0907', '0908', '0909', '0910', '0911', '0912',
     '1001', '1002', '1003', '1004', '1005', '1006', '1007', '1008', '1009', '1010', '1011', '1012',
     '1101', '1102', '1103', '1104', '1105', '1106', '1107', '1108', '1109', '1110', '1111', '1112',
     '1201', '1202', '1203', '1204', '1205', '1206', '1207', '1208', '1209', '1210', '1211', '1212',
     '1301', '1302', '1303', '1304', '1305', '1306', '1307', '1308', '1309', '1310', '1311', '1312',
     '1401', '1402', '1403', '1404', '1405', '1406', '1407', '1408', '1409', '1410', '1411', '1412',
     '1501', '1502', '1503', '1504', '1505', '1506', '1507', '1508', '1509', '1510', '1511', '1512',
     '1601', '1602', '1603', '1604', '1605', '1606', '1607', '1608', '1609', '1610', '1611', '1612'
]
year = '2009_2016'

list_data = []
for month in months:
    print month
    data = pd.read_csv('../data/energy/ae%s.csv' % month)

    # handle the header of data file
    firsts = ['time', 'not_care', 'not_care', 'maximum_amplitude', 'energy']
    hzs = []
    for i in range(22):
        hzs.append('hz' + str(i))
    not_care = ['not_care']*25
    rests = []
    for i in range(13):
        rests.append('rest' + str(i))
    header = firsts + hzs + not_care + rests
    data.columns = header
    needed_data = data[[x for x in header if x != 'not_care']].copy()
    list_data.append(needed_data)

all_data = pd.concat(list_data)
all_data.to_csv('../data/energy/data_%s.csv' % year, index=None)
