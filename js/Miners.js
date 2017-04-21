var field = [];
var showf = []; //unused
var H = 8;
var W = 8;
var global_var = 0;
var Number_of_bombs = 15;


zero_line = function(x) {
    var r = [];
    for (var i = 0; i < x; i++) {
        r.push(0);
    }
    return r;
}
for (var i = 0; i < H; i++) {
    field.push(zero_line(W));
    showf.push(zero_line(W));
}

test1 = function(v, x) { return Number(x == 'bomb') + v; };
test2 = function(v, x) { return x.reduce(test1, v) };
while (field.reduce(test2, 0) < Number_of_bombs) {
    var a = Math.round(Math.random() * 7);
    var b = Math.round(Math.random() * 7);
    field[a][b] = 'bomb';
}

function assign_func(sth, fld) {
    var td = document.getElementsByClassName(sth)[0];
    var elmnt = fld[Number(sth[1]) - 1][Number(sth[2]) - 1]
    td.ondblclick = function() {
        //console.log('this is cell: ' + sth);
        if (elmnt === 'bomb') {
            winning_loosing("Sorry!, Game Over.");
            //document.getElementById('res').innerText = ;
            //td.innerText = 'X';
        } else {
            var ind_2_uncover = [
                [Number(sth[1]) - 1, Number(sth[2]) - 1]
            ];
            var counter = 0;
            do {
                cells = check_8_nighbours(ind_2_uncover[counter][0], ind_2_uncover[counter][1]);
                counter += 1;
                for (var cel in cells) {
                    if (cells.hasOwnProperty(cel)) {
                        var element = cells[cel];
                        if ((-1 < element[0]) & (element[0] < H) & (-1 < element[1]) & (element[1] < W)) {
                            if (fld[element[0]][element[1]] === 0) {
                                ind_2_uncover.push([element[0], element[1]]);
                            }
                        }
                    }
                }
                //console.log(ind_2_uncover);
            }
            while ((ind_2_uncover.length > counter) & (counter < 3));
            for (var n = 0; n < ind_2_uncover.length; n++) {
                var x0 = ind_2_uncover[n];
                var x1 = fld[x0[0]][x0[1]];
                //console.log('c' + (x0[0] + 1) + (x0[1] + 1));
                var tds = document.getElementsByClassName('c' + (x0[0] + 1) + (x0[1] + 1))[0];

                if (!(tds.style.backgroundColor.length)) {
                    global_var += 1;
                    if (global_var > 62) {
                        winning_loosing('Hey, You Win!/and Waste time :(');
                    }
                }
                tds.style.backgroundColor = 'white';
                if (x1) {
                    tds.innerText = x1;
                    tds.style.padding = '1px';
                }
            }
        }
    };
    td.onclick = function() {
        //console.log(global_var);
        if (!(td.style.backgroundColor.length)) {
            td.style.backgroundColor = 'red';
            global_var += 1;
            if (global_var > 62) {
                winning_loosing('Hey, You Win!/and Waste time :(');
            }
        }
    }
}

for (var i = 1; i < (H + 1); i++) {
    for (var m = 1; m < (W + 1); m++) {
        cell_name = 'c' + i + m; //using the augmaning here
        field = update_field(i, m, field);
    }
}

for (var i = 1; i < (H + 1); i++) {
    //console.log('=======')
    for (var m = 1; m < (W + 1); m++) {
        cell_name = 'c' + i + m; //using the augmaning here
        assign_func(cell_name, field);
    }
}


function update_field(r, c, grid) {
    if (grid[r - 1][c - 1] == 'bomb') {
        return increase_suranded_by_1(r - 1, c - 1, grid);
    }
    return grid;
}

function increase_suranded_by_1(r, c, grid) {
    //console.log(grid)
    cells = check_8_nighbours(r, c);
    for (var cel in cells) {
        if (cells.hasOwnProperty(cel)) {
            var element = cells[cel];
            if ((-1 < element[0]) & (element[0] < H) & (-1 < element[1]) & (element[1] < W)) {
                if (typeof(grid[element[0]][element[1]]) == 'number') {
                    grid[element[0]][element[1]] += 1;
                }
            }
        }
    }
    return grid;
}

function check_8_nighbours(r, c) {
    var clls = [];
    for (var i = -1; i < 2; i++) {
        for (var z = -1; z < 2; z++) {
            if (i | z) {
                clls.push([r + i, c + z]);
            }
        }
    }
    return clls;
}

winning_loosing = function(msg) {
        var r = document.getElementById('res');
        r.style.color = 'red';
        r.style.textAlign = 'center';
        r.style.fontSize = '4em';
        r.innerText = msg;
    }
    //console.log(field);
