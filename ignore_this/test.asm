section .data
    msg db "Result: ", 0
    fmt db "%d", 0
    msg2 db "Sum of cubes: ", 0
    fmt2 db "%d", 0
    msg3 db "Final value: ", 0
    fmt3 db "%d", 0

section .bss
    result resd 1
    cubesum resd 1
    finalval resd 1

section .text
    global main
    extern printf

main:
    push ebp
    mov ebp, esp

    mov ecx, 10          ; loop counter
    mov eax, 0           ; accumulator for squares

complex_loop:
    mov edx, ecx
    imul edx, edx        ; edx = ecx * ecx
    add eax, edx         ; accumulate squares
    dec ecx
    jnz complex_loop

    mov [result], eax

    ; Print sum of squares
    push dword [result]
    push fmt
    call printf
    add esp, 8

    ; Calculate sum of cubes
    mov ecx, 10
    mov eax, 0

cube_loop:
    mov edx, ecx
    imul edx, edx        ; edx = ecx * ecx
    imul edx, ecx        ; edx = ecx * ecx * ecx
    add eax, edx         ; accumulate cubes
    dec ecx
    jnz cube_loop

    mov [cubesum], eax

    ; Print sum of cubes
    push dword [cubesum]
    push fmt2
    call printf
    add esp, 8

    ; Do some extra math
    mov eax, [result]
    mov edx, [cubesum]
    add eax, edx         ; sum squares and cubes
    mov ebx, 5
    imul eax, ebx        ; multiply by 5
    sub eax, 123         ; subtract 123
    mov [finalval], eax

    ; Print final value
    push dword [finalval]
    push fmt3
    call printf
    add esp, 8

    mov esp, ebp
    pop ebp
    ret