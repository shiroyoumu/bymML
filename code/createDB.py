import pandas as pd
import sqlite3 as lite

# 文件声明
pathDataset = "../data/dataset.csv"     # 数据集文件
pathDataDB = "../data/dataset_db.db"    # 数据库文件

hostList = ['host0001', 'host0021', 'host0027', 'host0029', 'host0030', 'host0035', 'host0039', 'host0041', 'host0049', 'host0056', 'host0057', 'host0063', 'host0070', 'host0073', 'host0079', 'host0081', 'host0118', 'host0131', 'host0143', 'host0150', 'host0167', 'host0171', 'host0175', 'host0194', 'host0196', 'host0204', 'host0210', 'host0218', 'host0228', 'host0239', 'host0261', 'host0279', 'host0281', 'host0290', 'host0291', 'host0294', 'host0336', 'host0337', 'host0350', 'host0354', 'host0363', 'host0372', 'host0374', 'host0378', 'host0380', 'host0384', 'host0391', 'host0408', 'host0445', 'host0447', 'host0449', 'host0454', 'host0467', 'host0474', 'host0492', 'host0496', 'host0500', 'host0509', 'host0517', 'host0518', 'host0534', 'host0547', 'host0548', 'host0562', 'host0567', 'host0573', 'host0591', 'host0593', 'host0609', 'host0622', 'host0629', 'host0636', 'host0637', 'host0648', 'host0649', 'host0658', 'host0689', 'host0696', 'host0703', 'host0708', 'host0711', 'host0733', 'host0746', 'host0769', 'host0788', 'host0803', 'host0805', 'host0837', 'host0840', 'host0861', 'host0877', 'host0884', 'host0898', 'host0899', 'host0911', 'host0914', 'host0950', 'host0959', 'host0964', 'host0970', 'host0985', 'host0986', 'host0994', 'host1004', 'host1006', 'host1017', 'host1021', 'host1037', 'host1042', 'host1046', 'host1057', 'host1084', 'host1099', 'host1105', 'host1115', 'host1120', 'host1121', 'host1126', 'host1131', 'host1136', 'host1146', 'host1149', 'host1153', 'host1157', 'host1177', 'host1189', 'host1205', 'host1207', 'host1217', 'host1225', 'host1249', 'host1258', 'host1276', 'host1296', 'host1325', 'host1330', 'host1345', 'host1367', 'host1372', 'host1411', 'host1416', 'host1443', 'host1474', 'host1489', 'host1494', 'host1497', 'host1501', 'host1503', 'host1505', 'host1524', 'host1548', 'host1567', 'host1570', 'host1601', 'host1626', 'host1627', 'host1635', 'host1638', 'host1663', 'host1695', 'host1696', 'host1705', 'host1712', 'host1721', 'host1731', 'host1748', 'host1762', 'host1763', 'host1770', 'host1785', 'host1788', 'host1794', 'host1809', 'host1821', 'host1837', 'host1847', 'host1850', 'host1872', 'host1896', 'host1938', 'host1979', 'host2013', 'host2023', 'host2029', 'host2073', 'host2082', 'host2089', 'host2109', 'host2118', 'host2122', 'host2134', 'host2142', 'host2145', 'host2147', 'host2150', 'host2159', 'host2163', 'host2168', 'host2178', 'host2179', 'host2202', 'host2205', 'host2218', 'host2225', 'host2226', 'host2248', 'host2269', 'host2273', 'host2284', 'host2296', 'host2307', 'host2310', 'host2311', 'host2315', 'host2317', 'host2325', 'host2341', 'host2343', 'host2350', 'host2357', 'host2358', 'host2363', 'host2365', 'host2375', 'host2376', 'host2381', 'host2383', 'host2384', 'host2402', 'host2404', 'host2408', 'host2438', 'host2462', 'host2466', 'host2493', 'host2498', 'host2521', 'host2532', 'host2536', 'host2553', 'host2559', 'host2565', 'host2567', 'host2585', 'host2605', 'host2616', 'host2617', 'host2619', 'host2635', 'host2636', 'host2643', 'host2659', 'host2671', 'host2673', 'host2674', 'host2682', 'host2702', 'host2706', 'host2714', 'host2717', 'host2732', 'host2751', 'host2754', 'host2757', 'host2765', 'host2776', 'host2780', 'host2789', 'host2790', 'host2798', 'host2802', 'host2833', 'host2841', 'host2853', 'host2862', 'host2871', 'host2875', 'host2879', 'host2897', 'host2901', 'host2913', 'host2920', 'host2936', 'host2953', 'host2994', 'host3000', 'host3001', 'host3013', 'host3022', 'host3031', 'host3033', 'host3041', 'host3069', 'host3086', 'host3103', 'host3117', 'host3128', 'host3140', 'host3141', 'host3160', 'host3163', 'host3168', 'host3203', 'host3208', 'host3209', 'host3232', 'host3249', 'host3253', 'host3272', 'host3322', 'host3326', 'host3332', 'host3333', 'host3375', 'host3378', 'host3381', 'host3401', 'host3402', 'host3403', 'host3407', 'host3426', 'host3451', 'host3464', 'host3467', 'host3470', 'host3478', 'host3512', 'host3537', 'host3539', 'host3546', 'host3557', 'host3564', 'host3587', 'host3606', 'host3632', 'host3634', 'host3641', 'host3651', 'host3670', 'host3673', 'host3677', 'host3688', 'host3691', 'host3692', 'host3695', 'host3707', 'host3727', 'host3729', 'host3742', 'host3757', 'host3762', 'host3774', 'host3776', 'host3782', 'host3806', 'host3841', 'host3866', 'host3867', 'host3873', 'host3894', 'host3895', 'host3897', 'host3898', 'host3906', 'host3914', 'host3931', 'host3936', 'host3939', 'host3957', 'host3958', 'host3962', 'host3980', 'host3996', 'host4033', 'host4038', 'host4050', 'host4054', 'host4064', 'host4085', 'host4086', 'host4091', 'host4093', 'host4111', 'host4119', 'host4130', 'host4133', 'host4148', 'host4157', 'host4180', 'host4188', 'host4203', 'host4211', 'host4225', 'host4231', 'host4248', 'host4249', 'host4259', 'host4280', 'host4286', 'host4289', 'host4293', 'host4325', 'host4331', 'host4336', 'host4340', 'host4348', 'host4363', 'host4372', 'host4384', 'host4397', 'host4415', 'host4419', 'host4423', 'host4426', 'host4451', 'host4458', 'host4460', 'host4461', 'host4464', 'host4473', 'host4488', 'host4489', 'host4491', 'host4493', 'host4512', 'host4513', 'host4516', 'host4524', 'host4533', 'host4545', 'host4556', 'host4561', 'host4562', 'host4568', 'host4569', 'host4575', 'host4585', 'host4610', 'host4631', 'host4637', 'host4641', 'host4652', 'host4668', 'host4680', 'host4684', 'host4687', 'host4696', 'host4702', 'host4703', 'host4713', 'host4725', 'host4767', 'host4768', 'host4770', 'host4771', 'host4784', 'host4785', 'host4788', 'host4817', 'host4828', 'host4832', 'host4846', 'host4847', 'host4856', 'host4881', 'host4887', 'host4916', 'host4944', 'host4945', 'host4956', 'host4967', 'host4975', 'host4980', 'host4989', 'host4999', 'host5005', 'host5025', 'host5026', 'host5036', 'host5064', 'host5073', 'host5075', 'host5084', 'host5089', 'host5092', 'host5099', 'host5112', 'host5117', 'host5125', 'host5151', 'host5163', 'host5190', 'host5199', 'host5219', 'host5243', 'host5245', 'host5259', 'host5281', 'host5307', 'host5314', 'host5323', 'host5327', 'host5343', 'host5344', 'host5355', 'host5365', 'host5373', 'host5392', 'host5396', 'host5398', 'host5404', 'host5410', 'host5415', 'host5418', 'host5421', 'host5431', 'host5434', 'host5437', 'host5438', 'host5442', 'host5445', 'host5452', 'host5468', 'host5479', 'host5481', 'host5485', 'host5499', 'host5513', 'host5518', 'host5521', 'host5523', 'host5524', 'host5525', 'host5556', 'host5572', 'host5575', 'host5601', 'host5603', 'host5609', 'host5624', 'host5631', 'host5632', 'host5637', 'host5641', 'host5647', 'host5656', 'host5657', 'host5663', 'host5666', 'host5673', 'host5674', 'host5705', 'host5737', 'host5744', 'host5746', 'host5762', 'host5781', 'host5797', 'host5820', 'host5834', 'host5841', 'host5855', 'host5857', 'host5867', 'host5913', 'host5941', 'host5942', 'host5968', 'host5974', 'host5992', 'host5997', 'host5998', 'host6016', 'host6032', 'host6036', 'host6041', 'host6050', 'host6056', 'host6072', 'host6080', 'host6089', 'host6102', 'host6115', 'host6117', 'host6120', 'host6126', 'host6128', 'host6139', 'host6176', 'host6190', 'host6205', 'host6222', 'host6227', 'host6238', 'host6255', 'host6274', 'host6282', 'host6291', 'host6295', 'host6300', 'host6304', 'host6319', 'host6323', 'host6325', 'host6329', 'host6332', 'host6357', 'host6367', 'host6370', 'host6371', 'host6378', 'host6381', 'host6395', 'host6397', 'host6398', 'host6411', 'host6435', 'host6462', 'host6467', 'host6494', 'host6495', 'host6500', 'host6517', 'host6525', 'host6528', 'host6529', 'host6533', 'host6541', 'host6548', 'host6554', 'host6556', 'host6571', 'host6579', 'host6583', 'host6590', 'host6591', 'host6601', 'host6602', 'host6615', 'host6623', 'host6647', 'host6663', 'host6667', 'host6668', 'host6671', 'host6672', 'host6685', 'host6707', 'host6709', 'host6720', 'host6723', 'host6724', 'host6727', 'host6741', 'host6742', 'host6745', 'host6768', 'host6809', 'host6817', 'host6848', 'host6853', 'host6865', 'host6869', 'host6883', 'host6889', 'host6903', 'host6917', 'host6919', 'host6922', 'host6930', 'host6933', 'host6934', 'host6959', 'host6968', 'host6973', 'host6985', 'host6991', 'host6997', 'host7043', 'host7045', 'host7058', 'host7060', 'host7074', 'host7083', 'host7087', 'host7090', 'host7110', 'host7117', 'host7119', 'host7121', 'host7126', 'host7186', 'host7191', 'host7192', 'host7195', 'host7198', 'host7200', 'host7209', 'host7217', 'host7227', 'host7233', 'host7246', 'host7255', 'host7267', 'host7319', 'host7320', 'host7326', 'host7332', 'host7371', 'host7373', 'host7375', 'host7380', 'host7381', 'host7389', 'host7396', 'host7401', 'host7402', 'host7412', 'host7425', 'host7430', 'host7431', 'host7440', 'host7462', 'host7468', 'host7475', 'host7481', 'host7494', 'host7510', 'host7513', 'host7514', 'host7516', 'host7523', 'host7524', 'host7528', 'host7570', 'host7576', 'host7582', 'host7589', 'host7591', 'host7620', 'host7624', 'host7635', 'host7637', 'host7639', 'host7648', 'host7658', 'host7694', 'host7700', 'host7704', 'host7712', 'host7726', 'host7735', 'host7738', 'host7745', 'host7757', 'host7764', 'host7771', 'host7774', 'host7788', 'host7791', 'host7816', 'host7822', 'host7828', 'host7844', 'host7846', 'host7849', 'host7851', 'host7866', 'host7883', 'host7893', 'host7896', 'host7907', 'host7908', 'host7920', 'host7921', 'host7924', 'host7936', 'host7946', 'host7956', 'host7964', 'host7986', 'host7993', 'host8008', 'host8028', 'host8030', 'host8040', 'host8045', 'host8062', 'host8063', 'host8064', 'host8068', 'host8078', 'host8086', 'host8099', 'host8100', 'host8135', 'host8143', 'host8157', 'host8178', 'host8179', 'host8197', 'host8235', 'host8236', 'host8254', 'host8258', 'host8259', 'host8287', 'host8292', 'host8298', 'host8314', 'host8315', 'host8330', 'host8343', 'host8362', 'host8374', 'host8377', 'host8401', 'host8402', 'host8409', 'host8427', 'host8428', 'host8449', 'host8456', 'host8490', 'host8500', 'host8502', 'host8509', 'host8521', 'host8522', 'host8536', 'host8543', 'host8544', 'host8569', 'host8574', 'host8577', 'host8585', 'host8599', 'host8609', 'host8611', 'host8622', 'host8636', 'host8652', 'host8670', 'host8682', 'host8704', 'host8726', 'host8746', 'host8747', 'host8757', 'host8759', 'host8762', 'host8778', 'host8790', 'host8791', 'host8794', 'host8797', 'host8802', 'host8805', 'host8808', 'host8815', 'host8824', 'host8849', 'host8857', 'host8863', 'host8881', 'host8883', 'host8884', 'host8902', 'host8920', 'host8921', 'host8923', 'host8934', 'host8963', 'host8972', 'host8980', 'host8986', 'host8990', 'host8996', 'host8999', 'host9011', 'host9024', 'host9025', 'host9035', 'host9047', 'host9060', 'host9065', 'host9067', 'host9077', 'host9084', 'host9088', 'host9098', 'host9106', 'host9107', 'host9124', 'host9128', 'host9138', 'host9150', 'host9151', 'host9156', 'host9161', 'host9165', 'host9200', 'host9202', 'host9220', 'host9222', 'host9225', 'host9227', 'host9229', 'host9242', 'host9256', 'host9259', 'host9261', 'host9263', 'host9274', 'host9277', 'host9278', 'host9297', 'host9327', 'host9330', 'host9333', 'host9337', 'host9345', 'host9355', 'host9364', 'host9374', 'host9375', 'host9380', 'host9381', 'host9398', 'host9402', 'host9418', 'host9425', 'host9428', 'host9437', 'host9439', 'host9441', 'host9442', 'host9448', 'host9453', 'host9460', 'host9474', 'host9478', 'host9485', 'host9487', 'host9496', 'host9502', 'host9508', 'host9519', 'host9552', 'host9555', 'host9558', 'host9564', 'host9588', 'host9590', 'host9604', 'host9605', 'host9610', 'host9625', 'host9632', 'host9650', 'host9674', 'host9676', 'host9684', 'host9685', 'host9689', 'host9693', 'host9698', 'host9720', 'host9733', 'host9742', 'host9748', 'host9761', 'host9766', 'host9773', 'host9774', 'host9783', 'host9803', 'host9812', 'host9842', 'host9846', 'host9863', 'host9868', 'host9873', 'host9888', 'host9897', 'host9906', 'host9926', 'host9936', 'host9951', 'host9957', 'host9958', 'host9961', 'host9981', 'host9988', 'host9992', 'host9993']
series = "cpuusagebyproc"

def scaling(df, scaling_upper_bound=100) -> (pd.DataFrame, float):
    '''
    对Mean数据进行缩放：将数据中最大值缩放至scaling_upper_bound

    :param df: 数据
    :param scaling_upper_bound: 缩放上界
    :return: 缩放后的数据
    '''
    maximum = df['Mean'].max()
    if maximum > 0:
        df['Mean'] = df['Mean'].apply(lambda x: (x / maximum) * scaling_upper_bound)
    return df, maximum

def preProcess(host, series) -> pd.DataFrame:
    '''
    对host主机中的series指标进行预处理：进行缩放、限定时间范围、补齐空缺
    :param host: 主机
    :param series: 指标
    :return: 预处理后的数据
    '''
    df = dataSet    # 读数据集
    df = df.loc[(df["hostname"] == host) & (df["series"] == series)]
    df, maxMean = scaling(df, 100)  # 对Mean缩放
    # 以时间戳作为下标，保证留出空缺时间空位
    df["time_window"] = pd.to_datetime(df["time_window"], format="%Y-%m-%d %H:%M:%")
    newIndex = pd.date_range("2019-12-03 00:00:00", "2020-02-10 23:00:00", freq="1H")
    df = df.set_index("time_window")
    df = df.reindex(newIndex)
    df = df.interpolate(method="linear", axis=0).ffill().bfill()
    df = df.reset_index()
    df.rename(columns={"index": "time_window"}, inplace=True)
    return df

if __name__ == '__main__':
    dataSet = pd.read_csv(pathDataset)  # 读数据集
    con = lite.connect(pathDataDB)
    for i in hostList:
        data = preProcess(i, series)
        data.to_sql("datasetDB", con, if_exists='append')
        print(i)

    con.close()


























    # con = lite.connect(pathDataDB)
    # data = pd.read_sql("select distinct(hostname) from datasetDB", con)


