#!/bin/bash

cd {{ directory }}

{{ command }} 2> proc.err 1> proc.out &
CMD_PID=$!
echo $CMD_PID
