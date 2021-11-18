import matplotlib.pyplot as plt
import fpmodules as fp

# Bumblebee sessions (Bombus terrestris)
# ID 973 - 8038 events
# ID 1253 - 2457 events
# ID 1289 - 3269 events
# ID 1301 - 8226 events
# ID 1307 - 7062 events

# session 457 (male mosquitoes) with 10104 events
# If male mosquitoes don't sell:
# session 459 (mix of male and female mosquitoes) with 11364

labelled_bumblebee_query = "select top 10 * from measurement where sessionid=973"
labelled_male_mosquito_query = "select top 10 * from measurement where sessionid=457"
labelled_male_and_female_mosquito_query = "select top 10 * from measurement where sessionid=459"
df = fp.dbquery(labelled_bumblebee_query)
df.head()

event = fp.Event(df['Id'][0])
print(event)

event.fill()

# print(event.data)
# event.as_dataframe()

# event.plot()
# plt.show()


event_list = fp.EventList(df['Id'].tolist()).fill()
event_list.plot()
plt.show()
