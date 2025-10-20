# Mpi Prompt List

This node is the base of the Mpi Prompt Gen system.

It converts a text into a list based on new lines and randomly shuffles each item. 

## Parameters

- **title**: The title used to trigger this list in a Prompt Processor
- **options**: The list of options for this this node

## Usage

Connect to a prompt processor and access its outputs using camel case wording.

Let's consider the list you are trying to trigger is named "locations" and has the following options:

- Car Park
- Movie Theater
- Forest

In a prompt processor you can trigger an option on this list in 3 ways:
- ***Single item***: \_locations_

This will trigger a single item from your list like: 
```
I went to the _locations_
Possible Output: I went to the Forest
```

- ***Multiple items***: \_locations_x3_

This will trigger 3 items from your list like: 

```
Various locations like the _locations_x3_
Possible Output: Various locations like the Movie Theater, Car Park, Forest
```

- ***Single pick***: \_locations_2_

This will trigger option 2 from your list, useful for when you want to trigger more than 1 item:

```
I went to the _locations_1_ and to the _locations_2_.
Possible Output: I went to the Forest and to the Car Park.
```
