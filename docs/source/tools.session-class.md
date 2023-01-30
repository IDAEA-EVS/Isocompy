# class session ( )

The class to save and load the objects and sessions

---


## **Methods**

* save()

* load()

* save_session()

* load_session()

---


## session.save ( )

session.save(`self,name="isocompy_saved_object"`)

The method to save an object

---


### **Parameters**

**name** str default=`"isocompy_saved_object"`

The output name string

---


### **Returns**

**filename** string

Directory of the saved object

---


## session.load ( )

session.load(`direc`)

The method to load a pkl object. `direc` is the directory of the object to be loaded.

---


### **Returns**

**obj** object

The loaded object

---


## session.save_session ( )

session.save_session(`direc, name="isocompy_saved_session", *argv`):

The method to save a session

---


### **Parameters**

name: str default="isocompy_saved_object"

The output name string

---
**\*argv**

The objects that wanted to be stored in the session

---


### **Returns**

**filename** string 

Directory of the saved session

---


## session.load_session ( )

session.load_session(`dir`)

The method to load a session

---


### **Parameters**

**\*argv**

The objects that wanted to be stored in the session

---


### **Returns**

Loads the session

