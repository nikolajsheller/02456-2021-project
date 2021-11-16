import fpmodules as fp
import matplotlib.pyplot as plt

query = "select top 10 * from measurement where sessionid=385"
df = fp.dbquery(query)
df.head()

event = fp.Event(df['Id'][0])
print(event)

event.fill()

print(event.data)
event.as_dataframe()

event.plot();
plt.show()

event_list = fp.EventList(df['Id'].tolist()).fill();
event_list.plot();
plt.show()


# raw data
mac_address = fp.get_macaddress(name=20)
data, times, ds = fp.get_raw_data(mac_address, dateid=20210215, starttimeid=85000, endtimeid=90000);
plt.figure(figsize=(12,5))
ax = fp.plot_raw_data(data=data, times=times, ds=ds, ax=plt.gca())
plt.show()