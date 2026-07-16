# Event-grounded ALFWorld execution policy

Convert the task into an ordered checklist before acting:

1. Identify the exact target object type and required quantity.
2. Identify every required state change, such as clean, heat, cool, slice, or
   illuminate while holding the object.
3. Identify the final destination.
4. Execute prerequisites in this order: locate the target, take it, perform the
   required state change, then place or inspect it at the destination.

Treat the available action list as the authoritative state. A transformation or
placement action is executable only when its exact action appears there. If it
does not appear, satisfy the missing prerequisite instead of going repeatedly to
the appliance or destination.

## Locate and take

- At the start, search for the target before visiting its appliance or final
  destination. Do not assume that the target is already in the fridge,
  microwave, sink, or destination.
- Search visible surfaces first when their observation lists the target. For
  closed receptacles, go to one candidate, open it, inspect the contents, and
  move on if the target is absent.
- Do not close an empty receptacle merely to tidy it; the step budget is scarce.
  Do not revisit a receptacle whose contents were already checked.
- When the target becomes visible and a `take ... from ...` action is available,
  take it immediately. Use `inventory` when uncertain whether the target is
  already held.
- For tasks requiring two objects, finish one object's full checklist and then
  repeat for the second. Remember which required instances are already placed.

## Apply required events

- Clean: while holding the target, go to the sink basin and use the available
  `clean TARGET with sinkbasin` action.
- Heat: while holding the target, go to the microwave, open it if required, put
  the target inside if required by the available actions, and use the exact heat
  action. Recover the target afterward before carrying it to the destination.
- Cool: while holding the target, go to the fridge, open it if required, and use
  the exact cool action. Recover the target afterward if the environment puts it
  inside the fridge.
- Slice: obtain and hold a knife before attempting to slice the target. Then go
  to the target and use the exact slice action.
- Light/inspect: obtain the requested object first when the goal says to examine
  it under a light or to hold it while turning on a light. Go to the lamp only
  after that prerequisite, turn it on, then examine the target if required.

Never substitute one event for another. Words such as wet/rinsed mean clean,
hot/microwaved mean heat, and cold/chilled mean cool.

## Finish

- After all required state changes are complete and the target is held, go to
  the named destination. Open it only if the exact placement action requires an
  open receptacle, then use the available `put ... in/on ...` action.
- Do not stop after reaching the destination or appliance. Success requires the
  final placement or examination action and the environment's completion signal.
- If an attempted action is unavailable or invalid, read the current available
  actions and repair the missing prerequisite. Do not repeat the same invalid
  command or restart the plan from its first step.
