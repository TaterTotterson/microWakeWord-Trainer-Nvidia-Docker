//#region node_modules/@vue/shared/dist/shared.esm-bundler.js
// @__NO_SIDE_EFFECTS__
function e(e) {
	let t = /* @__PURE__ */ Object.create(null);
	for (let n of e.split(",")) t[n] = 1;
	return (e) => e in t;
}
var t = {}, n = [], r = () => {}, i = () => !1, a = (e) => e.charCodeAt(0) === 111 && e.charCodeAt(1) === 110 && (e.charCodeAt(2) > 122 || e.charCodeAt(2) < 97), o = (e) => e.startsWith("onUpdate:"), s = Object.assign, c = (e, t) => {
	let n = e.indexOf(t);
	n > -1 && e.splice(n, 1);
}, l = Object.prototype.hasOwnProperty, u = (e, t) => l.call(e, t), d = Array.isArray, f = (e) => x(e) === "[object Map]", p = (e) => x(e) === "[object Set]", m = (e) => x(e) === "[object Date]", h = (e) => typeof e == "function", g = (e) => typeof e == "string", _ = (e) => typeof e == "symbol", v = (e) => typeof e == "object" && !!e, y = (e) => (v(e) || h(e)) && h(e.then) && h(e.catch), b = Object.prototype.toString, x = (e) => b.call(e), S = (e) => x(e).slice(8, -1), C = (e) => x(e) === "[object Object]", w = (e) => g(e) && e !== "NaN" && e[0] !== "-" && "" + parseInt(e, 10) === e, T = /* @__PURE__ */ e(",key,ref,ref_for,ref_key,onVnodeBeforeMount,onVnodeMounted,onVnodeBeforeUpdate,onVnodeUpdated,onVnodeBeforeUnmount,onVnodeUnmounted"), ee = (e) => {
	let t = /* @__PURE__ */ Object.create(null);
	return ((n) => t[n] || (t[n] = e(n)));
}, te = /-\w/g, E = ee((e) => e.replace(te, (e) => e.slice(1).toUpperCase())), ne = /\B([A-Z])/g, D = ee((e) => e.replace(ne, "-$1").toLowerCase()), re = ee((e) => e.charAt(0).toUpperCase() + e.slice(1)), ie = ee((e) => e ? `on${re(e)}` : ""), ae = (e, t) => !Object.is(e, t), oe = (e, ...t) => {
	for (let n = 0; n < e.length; n++) e[n](...t);
}, O = (e, t, n, r = !1) => {
	Object.defineProperty(e, t, {
		configurable: !0,
		enumerable: !1,
		writable: r,
		value: n
	});
}, se = (e) => {
	let t = parseFloat(e);
	return isNaN(t) ? e : t;
}, ce = (e) => {
	let t = g(e) ? Number(e) : NaN;
	return isNaN(t) ? e : t;
}, le, ue = () => le ||= typeof globalThis < "u" ? globalThis : typeof self < "u" ? self : typeof window < "u" ? window : typeof global < "u" ? global : {};
function de(e) {
	if (d(e)) {
		let t = {};
		for (let n = 0; n < e.length; n++) {
			let r = e[n], i = g(r) ? he(r) : de(r);
			if (i) for (let e in i) t[e] = i[e];
		}
		return t;
	}
	if (g(e) || v(e)) return e;
}
var fe = /;(?![^(]*\))/g, pe = /:([^]+)/, me = /\/\*[^]*?\*\//g;
function he(e) {
	let t = {};
	return e.replace(me, "").split(fe).forEach((e) => {
		if (e) {
			let n = e.split(pe);
			n.length > 1 && (t[n[0].trim()] = n[1].trim());
		}
	}), t;
}
function k(e) {
	let t = "";
	if (g(e)) t = e;
	else if (d(e)) for (let n = 0; n < e.length; n++) {
		let r = k(e[n]);
		r && (t += r + " ");
	}
	else if (v(e)) for (let n in e) e[n] && (t += n + " ");
	return t.trim();
}
var ge = "itemscope,allowfullscreen,formnovalidate,ismap,nomodule,novalidate,readonly", _e = /* @__PURE__ */ e(ge);
ge + "";
function ve(e) {
	return !!e || e === "";
}
function ye(e, t) {
	if (e.length !== t.length) return !1;
	let n = !0;
	for (let r = 0; n && r < e.length; r++) n = be(e[r], t[r]);
	return n;
}
function be(e, t) {
	if (e === t) return !0;
	let n = m(e), r = m(t);
	if (n || r) return n && r ? e.getTime() === t.getTime() : !1;
	if (n = _(e), r = _(t), n || r) return e === t;
	if (n = d(e), r = d(t), n || r) return n && r ? ye(e, t) : !1;
	if (n = v(e), r = v(t), n || r) {
		if (!n || !r || Object.keys(e).length !== Object.keys(t).length) return !1;
		for (let n in e) {
			let r = e.hasOwnProperty(n), i = t.hasOwnProperty(n);
			if (r && !i || !r && i || !be(e[n], t[n])) return !1;
		}
	}
	return String(e) === String(t);
}
function xe(e, t) {
	return e.findIndex((e) => be(e, t));
}
var Se = (e) => !!(e && e.__v_isRef === !0), A = (e) => g(e) ? e : e == null ? "" : d(e) || v(e) && (e.toString === b || !h(e.toString)) ? Se(e) ? A(e.value) : JSON.stringify(e, Ce, 2) : String(e), Ce = (e, t) => Se(t) ? Ce(e, t.value) : f(t) ? { [`Map(${t.size})`]: [...t.entries()].reduce((e, [t, n], r) => (e[we(t, r) + " =>"] = n, e), {}) } : p(t) ? { [`Set(${t.size})`]: [...t.values()].map((e) => we(e)) } : _(t) ? we(t) : v(t) && !d(t) && !C(t) ? String(t) : t, we = (e, t = "") => _(e) ? `Symbol(${e.description ?? t})` : e, j, Te = class {
	constructor(e = !1) {
		this.detached = e, this._active = !0, this._on = 0, this.effects = [], this.cleanups = [], this._isPaused = !1, this._warnOnRun = !0, this.__v_skip = !0, !e && j && (j.active ? (this.parent = j, this.index = (j.scopes || (j.scopes = [])).push(this) - 1) : (this._active = !1, this._warnOnRun = !1));
	}
	get active() {
		return this._active;
	}
	pause() {
		if (this._active) {
			this._isPaused = !0;
			let e, t;
			if (this.scopes) {
				let n = this.scopes.slice();
				for (e = 0, t = n.length; e < t; e++) n[e].pause();
			}
			for (e = 0, t = this.effects.length; e < t; e++) this.effects[e].pause();
		}
	}
	resume() {
		if (this._active && this._isPaused) {
			this._isPaused = !1;
			let e, t;
			if (this.scopes) {
				let n = this.scopes.slice();
				for (e = 0, t = n.length; e < t; e++) n[e].resume();
			}
			let n = this.effects.slice();
			for (e = 0, t = n.length; e < t; e++) n[e].resume();
		}
	}
	run(e) {
		if (this._active) {
			let t = j;
			try {
				return j = this, e();
			} finally {
				j = t;
			}
		}
	}
	on() {
		++this._on === 1 && (this.prevScope = j, j = this);
	}
	off() {
		if (this._on > 0 && --this._on === 0) {
			if (j === this) j = this.prevScope;
			else {
				let e = j;
				for (; e;) {
					if (e.prevScope === this) {
						e.prevScope = this.prevScope;
						break;
					}
					e = e.prevScope;
				}
			}
			this.prevScope = void 0;
		}
	}
	stop(e) {
		if (this._active) {
			this._active = !1;
			let t, n;
			for (t = 0, n = this.effects.length; t < n; t++) this.effects[t].stop();
			for (this.effects.length = 0, t = 0, n = this.cleanups.length; t < n; t++) this.cleanups[t]();
			if (this.cleanups.length = 0, this.scopes) {
				let e = this.scopes.slice();
				for (t = 0, n = e.length; t < n; t++) e[t].stop(!0);
				this.scopes.length = 0;
			}
			if (!this.detached && this.parent && !e) {
				let e = this.parent.scopes.pop();
				e && e !== this && (this.parent.scopes[this.index] = e, e.index = this.index);
			}
			this.parent = void 0;
		}
	}
};
function Ee() {
	return j;
}
var M, De = /* @__PURE__ */ new WeakSet(), Oe = class {
	constructor(e) {
		this.fn = e, this.deps = void 0, this.depsTail = void 0, this.flags = 5, this.next = void 0, this.cleanup = void 0, this.scheduler = void 0, j && (j.active ? j.effects.push(this) : this.flags &= -2);
	}
	pause() {
		this.flags |= 64;
	}
	resume() {
		this.flags & 64 && (this.flags &= -65, De.has(this) && (De.delete(this), this.trigger()));
	}
	notify() {
		this.flags & 2 && !(this.flags & 32) || this.flags & 8 || Me(this);
	}
	run() {
		if (!(this.flags & 1)) return this.fn();
		this.flags |= 2, Ge(this), Fe(this);
		let e = M, t = Ve;
		M = this, Ve = !0;
		try {
			return this.fn();
		} finally {
			Ie(this), M = e, Ve = t, this.flags &= -3;
		}
	}
	stop() {
		if (this.flags & 1) {
			for (let e = this.deps; e; e = e.nextDep) ze(e);
			this.deps = this.depsTail = void 0, Ge(this), this.onStop && this.onStop(), this.flags &= -2;
		}
	}
	trigger() {
		this.flags & 64 ? De.add(this) : this.scheduler ? this.scheduler() : this.runIfDirty();
	}
	runIfDirty() {
		Le(this) && this.run();
	}
	get dirty() {
		return Le(this);
	}
}, ke = 0, Ae, je;
function Me(e, t = !1) {
	if (e.flags |= 8, t) {
		e.next = je, je = e;
		return;
	}
	e.next = Ae, Ae = e;
}
function Ne() {
	ke++;
}
function Pe() {
	if (--ke > 0) return;
	if (je) {
		let e = je;
		for (je = void 0; e;) {
			let t = e.next;
			e.next = void 0, e.flags &= -9, e = t;
		}
	}
	let e;
	for (; Ae;) {
		let t = Ae;
		for (Ae = void 0; t;) {
			let n = t.next;
			if (t.next = void 0, t.flags &= -9, t.flags & 1) try {
				t.trigger();
			} catch (t) {
				e ||= t;
			}
			t = n;
		}
	}
	if (e) throw e;
}
function Fe(e) {
	for (let t = e.deps; t; t = t.nextDep) t.version = -1, t.prevActiveLink = t.dep.activeLink, t.dep.activeLink = t;
}
function Ie(e) {
	let t, n = e.depsTail, r = n;
	for (; r;) {
		let e = r.prevDep;
		r.version === -1 ? (r === n && (n = e), ze(r), Be(r)) : t = r, r.dep.activeLink = r.prevActiveLink, r.prevActiveLink = void 0, r = e;
	}
	e.deps = t, e.depsTail = n;
}
function Le(e) {
	for (let t = e.deps; t; t = t.nextDep) if (t.dep.version !== t.version || t.dep.computed && (Re(t.dep.computed) || t.dep.version !== t.version)) return !0;
	return !!e._dirty;
}
function Re(e) {
	if (e.flags & 4 && !(e.flags & 16) || (e.flags &= -17, e.globalVersion === Ke) || (e.globalVersion = Ke, !e.isSSR && e.flags & 128 && (!e.deps && !e._dirty || !Le(e)))) return;
	e.flags |= 2;
	let t = e.dep, n = M, r = Ve;
	M = e, Ve = !0;
	try {
		Fe(e);
		let n = e.fn(e._value);
		(t.version === 0 || ae(n, e._value)) && (e.flags |= 128, e._value = n, t.version++);
	} catch (e) {
		throw t.version++, e;
	} finally {
		M = n, Ve = r, Ie(e), e.flags &= -3;
	}
}
function ze(e, t = !1) {
	let { dep: n, prevSub: r, nextSub: i } = e;
	if (r && (r.nextSub = i, e.prevSub = void 0), i && (i.prevSub = r, e.nextSub = void 0), n.subs === e && (n.subs = r, !r && n.computed)) {
		n.computed.flags &= -5;
		for (let e = n.computed.deps; e; e = e.nextDep) ze(e, !0);
	}
	!t && !--n.sc && n.map && n.map.delete(n.key);
}
function Be(e) {
	let { prevDep: t, nextDep: n } = e;
	t && (t.nextDep = n, e.prevDep = void 0), n && (n.prevDep = t, e.nextDep = void 0);
}
var Ve = !0, He = [];
function Ue() {
	He.push(Ve), Ve = !1;
}
function We() {
	let e = He.pop();
	Ve = e === void 0 || e;
}
function Ge(e) {
	let { cleanup: t } = e;
	if (e.cleanup = void 0, t) {
		let e = M;
		M = void 0;
		try {
			t();
		} finally {
			M = e;
		}
	}
}
var Ke = 0, qe = class {
	constructor(e, t) {
		this.sub = e, this.dep = t, this.version = t.version, this.nextDep = this.prevDep = this.nextSub = this.prevSub = this.prevActiveLink = void 0;
	}
}, Je = class {
	constructor(e) {
		this.computed = e, this.version = 0, this.activeLink = void 0, this.subs = void 0, this.map = void 0, this.key = void 0, this.sc = 0, this.__v_skip = !0;
	}
	track(e) {
		if (!M || !Ve || M === this.computed) return;
		let t = this.activeLink;
		if (t === void 0 || t.sub !== M) t = this.activeLink = new qe(M, this), M.deps ? (t.prevDep = M.depsTail, M.depsTail.nextDep = t, M.depsTail = t) : M.deps = M.depsTail = t, Ye(t);
		else if (t.version === -1 && (t.version = this.version, t.nextDep)) {
			let e = t.nextDep;
			e.prevDep = t.prevDep, t.prevDep && (t.prevDep.nextDep = e), t.prevDep = M.depsTail, t.nextDep = void 0, M.depsTail.nextDep = t, M.depsTail = t, M.deps === t && (M.deps = e);
		}
		return t;
	}
	trigger(e) {
		this.version++, Ke++, this.notify(e);
	}
	notify(e) {
		Ne();
		try {
			for (let e = this.subs; e; e = e.prevSub) e.sub.notify() && e.sub.dep.notify();
		} finally {
			Pe();
		}
	}
};
function Ye(e) {
	if (e.dep.sc++, e.sub.flags & 4) {
		let t = e.dep.computed;
		if (t && !e.dep.subs) {
			t.flags |= 20;
			for (let e = t.deps; e; e = e.nextDep) Ye(e);
		}
		let n = e.dep.subs;
		n !== e && (e.prevSub = n, n && (n.nextSub = e)), e.dep.subs = e;
	}
}
var Xe = /* @__PURE__ */ new WeakMap(), Ze = /* @__PURE__ */ Symbol(""), Qe = /* @__PURE__ */ Symbol(""), $e = /* @__PURE__ */ Symbol("");
function N(e, t, n) {
	if (Ve && M) {
		let t = Xe.get(e);
		t || Xe.set(e, t = /* @__PURE__ */ new Map());
		let r = t.get(n);
		r || (t.set(n, r = new Je()), r.map = t, r.key = n), r.track();
	}
}
function et(e, t, n, r, i, a) {
	let o = Xe.get(e);
	if (!o) {
		Ke++;
		return;
	}
	let s = (e) => {
		e && e.trigger();
	};
	if (Ne(), t === "clear") o.forEach(s);
	else {
		let i = d(e), a = i && w(n);
		if (i && n === "length") {
			let e = Number(r);
			o.forEach((t, n) => {
				(n === "length" || n === $e || !_(n) && n >= e) && s(t);
			});
		} else switch ((n !== void 0 || o.has(void 0)) && s(o.get(n)), a && s(o.get($e)), t) {
			case "add":
				i ? a && s(o.get("length")) : (s(o.get(Ze)), f(e) && s(o.get(Qe)));
				break;
			case "delete":
				i || (s(o.get(Ze)), f(e) && s(o.get(Qe)));
				break;
			case "set": f(e) && s(o.get(Ze));
		}
	}
	Pe();
}
function tt(e) {
	let t = /* @__PURE__ */ P(e);
	return t === e ? t : (N(t, "iterate", $e), /* @__PURE__ */ Bt(e) ? t : t.map(Ut));
}
function nt(e) {
	return N(e = /* @__PURE__ */ P(e), "iterate", $e), e;
}
function rt(e, t) {
	return /* @__PURE__ */ zt(e) ? Wt(/* @__PURE__ */ Rt(e) ? Ut(t) : t) : Ut(t);
}
var it = {
	__proto__: null,
	[Symbol.iterator]() {
		return at(this, Symbol.iterator, (e) => rt(this, e));
	},
	concat(...e) {
		return tt(this).concat(...e.map((e) => d(e) ? tt(e) : e));
	},
	entries() {
		return at(this, "entries", (e) => (e[1] = rt(this, e[1]), e));
	},
	every(e, t) {
		return st(this, "every", e, t, void 0, arguments);
	},
	filter(e, t) {
		return st(this, "filter", e, t, (e) => e.map((e) => rt(this, e)), arguments);
	},
	find(e, t) {
		return st(this, "find", e, t, (e) => rt(this, e), arguments);
	},
	findIndex(e, t) {
		return st(this, "findIndex", e, t, void 0, arguments);
	},
	findLast(e, t) {
		return st(this, "findLast", e, t, (e) => rt(this, e), arguments);
	},
	findLastIndex(e, t) {
		return st(this, "findLastIndex", e, t, void 0, arguments);
	},
	forEach(e, t) {
		return st(this, "forEach", e, t, void 0, arguments);
	},
	includes(...e) {
		return lt(this, "includes", e);
	},
	indexOf(...e) {
		return lt(this, "indexOf", e);
	},
	join(e) {
		return tt(this).join(e);
	},
	lastIndexOf(...e) {
		return lt(this, "lastIndexOf", e);
	},
	map(e, t) {
		return st(this, "map", e, t, void 0, arguments);
	},
	pop() {
		return ut(this, "pop");
	},
	push(...e) {
		return ut(this, "push", e);
	},
	reduce(e, ...t) {
		return ct(this, "reduce", e, t);
	},
	reduceRight(e, ...t) {
		return ct(this, "reduceRight", e, t);
	},
	shift() {
		return ut(this, "shift");
	},
	some(e, t) {
		return st(this, "some", e, t, void 0, arguments);
	},
	splice(...e) {
		return ut(this, "splice", e);
	},
	toReversed() {
		return tt(this).toReversed();
	},
	toSorted(e) {
		return tt(this).toSorted(e);
	},
	toSpliced(...e) {
		return tt(this).toSpliced(...e);
	},
	unshift(...e) {
		return ut(this, "unshift", e);
	},
	values() {
		return at(this, "values", (e) => rt(this, e));
	}
};
function at(e, t, n) {
	let r = nt(e), i = r[t]();
	return r !== e && !/* @__PURE__ */ Bt(e) && (i._next = i.next, i.next = () => {
		let e = i._next();
		return e.done || (e.value = n(e.value)), e;
	}), i;
}
var ot = Array.prototype;
function st(e, t, n, r, i, a) {
	let o = nt(e), s = o !== e && !/* @__PURE__ */ Bt(e), c = o[t];
	if (c !== ot[t]) {
		let t = c.apply(e, a);
		return s ? Ut(t) : t;
	}
	let l = n;
	o !== e && (s ? l = function(t, r) {
		return n.call(this, rt(e, t), r, e);
	} : n.length > 2 && (l = function(t, r) {
		return n.call(this, t, r, e);
	}));
	let u = c.call(o, l, r);
	return s && i ? i(u) : u;
}
function ct(e, t, n, r) {
	let i = nt(e), a = i !== e && !/* @__PURE__ */ Bt(e), o = n, s = !1;
	i !== e && (a ? (s = r.length === 0, o = function(t, r, i) {
		return s && (s = !1, t = rt(e, t)), n.call(this, t, rt(e, r), i, e);
	}) : n.length > 3 && (o = function(t, r, i) {
		return n.call(this, t, r, i, e);
	}));
	let c = i[t](o, ...r);
	return s ? rt(e, c) : c;
}
function lt(e, t, n) {
	let r = /* @__PURE__ */ P(e);
	N(r, "iterate", $e);
	let i = r[t](...n);
	return (i === -1 || i === !1) && /* @__PURE__ */ Vt(n[0]) ? (n[0] = /* @__PURE__ */ P(n[0]), r[t](...n)) : i;
}
function ut(e, t, n = []) {
	Ue(), Ne();
	let r = (/* @__PURE__ */ P(e))[t].apply(e, n);
	return Pe(), We(), r;
}
var dt = /* @__PURE__ */ e("__proto__,__v_isRef,__isVue"), ft = new Set(/* @__PURE__ */ Object.getOwnPropertyNames(Symbol).filter((e) => e !== "arguments" && e !== "caller").map((e) => Symbol[e]).filter(_));
function pt(e) {
	_(e) || (e = String(e));
	let t = /* @__PURE__ */ P(this);
	return N(t, "has", e), t.hasOwnProperty(e);
}
var mt = class {
	constructor(e = !1, t = !1) {
		this._isReadonly = e, this._isShallow = t;
	}
	get(e, t, n) {
		if (t === "__v_skip") return e.__v_skip;
		let r = this._isReadonly, i = this._isShallow;
		if (t === "__v_isReactive") return !r;
		if (t === "__v_isReadonly") return r;
		if (t === "__v_isShallow") return i;
		if (t === "__v_raw") return n === (r ? i ? Mt : jt : i ? At : kt).get(e) || Object.getPrototypeOf(e) === Object.getPrototypeOf(n) ? e : void 0;
		let a = d(e);
		if (!r) {
			let e;
			if (a && (e = it[t])) return e;
			if (t === "hasOwnProperty") return pt;
		}
		let o = Reflect.get(e, t, /* @__PURE__ */ F(e) ? e : n);
		if ((_(t) ? ft.has(t) : dt(t)) || (r || N(e, "get", t), i)) return o;
		if (/* @__PURE__ */ F(o)) {
			let e = a && w(t) ? o : o.value;
			return r && v(e) ? /* @__PURE__ */ It(e) : e;
		}
		return v(o) ? r ? /* @__PURE__ */ It(o) : /* @__PURE__ */ Pt(o) : o;
	}
}, ht = class extends mt {
	constructor(e = !1) {
		super(!1, e);
	}
	set(e, t, n, r) {
		let i = e[t], a = d(e) && w(t);
		if (!this._isShallow) {
			let e = /* @__PURE__ */ zt(i);
			if (!/* @__PURE__ */ Bt(n) && !/* @__PURE__ */ zt(n) && (i = /* @__PURE__ */ P(i), n = /* @__PURE__ */ P(n)), !a && /* @__PURE__ */ F(i) && !/* @__PURE__ */ F(n)) return e || (i.value = n), !0;
		}
		let o = a ? Number(t) < e.length : u(e, t), s = Reflect.set(e, t, n, /* @__PURE__ */ F(e) ? e : r);
		return e === /* @__PURE__ */ P(r) && s && (o ? ae(n, i) && et(e, "set", t, n, i) : et(e, "add", t, n)), s;
	}
	deleteProperty(e, t) {
		let n = u(e, t), r = e[t], i = Reflect.deleteProperty(e, t);
		return i && n && et(e, "delete", t, void 0, r), i;
	}
	has(e, t) {
		let n = Reflect.has(e, t);
		return (!_(t) || !ft.has(t)) && N(e, "has", t), n;
	}
	ownKeys(e) {
		return N(e, "iterate", d(e) ? "length" : Ze), Reflect.ownKeys(e);
	}
}, gt = class extends mt {
	constructor(e = !1) {
		super(!0, e);
	}
	set(e, t) {
		return !0;
	}
	deleteProperty(e, t) {
		return !0;
	}
}, _t = /* @__PURE__ */ new ht(), vt = /* @__PURE__ */ new gt(), yt = /* @__PURE__ */ new ht(!0), bt = (e) => e, xt = (e) => Reflect.getPrototypeOf(e);
function St(e, t, n) {
	return function(...r) {
		let i = this.__v_raw, a = /* @__PURE__ */ P(i), o = f(a), c = e === "entries" || e === Symbol.iterator && o, l = e === "keys" && o, u = i[e](...r), d = n ? bt : t ? Wt : Ut;
		return !t && N(a, "iterate", l ? Qe : Ze), s(Object.create(u), { next() {
			let { value: e, done: t } = u.next();
			return t ? {
				value: e,
				done: t
			} : {
				value: c ? [d(e[0]), d(e[1])] : d(e),
				done: t
			};
		} });
	};
}
function Ct(e) {
	return function(...t) {
		return e === "delete" ? !1 : e === "clear" ? void 0 : this;
	};
}
function wt(e, t) {
	let n = {
		get(n) {
			let r = this.__v_raw, i = /* @__PURE__ */ P(r), a = /* @__PURE__ */ P(n);
			e || (ae(n, a) && N(i, "get", n), N(i, "get", a));
			let { has: o } = xt(i), s = t ? bt : e ? Wt : Ut;
			if (o.call(i, n)) return s(r.get(n));
			if (o.call(i, a)) return s(r.get(a));
			r !== i && r.get(n);
		},
		get size() {
			let t = this.__v_raw;
			return !e && N(/* @__PURE__ */ P(t), "iterate", Ze), t.size;
		},
		has(t) {
			let n = this.__v_raw, r = /* @__PURE__ */ P(n), i = /* @__PURE__ */ P(t);
			return e || (ae(t, i) && N(r, "has", t), N(r, "has", i)), t === i ? n.has(t) : n.has(t) || n.has(i);
		},
		forEach(n, r) {
			let i = this, a = i.__v_raw, o = /* @__PURE__ */ P(a), s = t ? bt : e ? Wt : Ut;
			return !e && N(o, "iterate", Ze), a.forEach((e, t) => n.call(r, s(e), s(t), i));
		}
	};
	return s(n, e ? {
		add: Ct("add"),
		set: Ct("set"),
		delete: Ct("delete"),
		clear: Ct("clear")
	} : {
		add(e) {
			let n = /* @__PURE__ */ P(this), r = xt(n), i = /* @__PURE__ */ P(e), a = !t && !/* @__PURE__ */ Bt(e) && !/* @__PURE__ */ zt(e) ? i : e;
			return r.has.call(n, a) || ae(e, a) && r.has.call(n, e) || ae(i, a) && r.has.call(n, i) || (n.add(a), et(n, "add", a, a)), this;
		},
		set(e, n) {
			!t && !/* @__PURE__ */ Bt(n) && !/* @__PURE__ */ zt(n) && (n = /* @__PURE__ */ P(n));
			let r = /* @__PURE__ */ P(this), { has: i, get: a } = xt(r), o = i.call(r, e);
			o ||= (e = /* @__PURE__ */ P(e), i.call(r, e));
			let s = a.call(r, e);
			return r.set(e, n), o ? ae(n, s) && et(r, "set", e, n, s) : et(r, "add", e, n), this;
		},
		delete(e) {
			let t = /* @__PURE__ */ P(this), { has: n, get: r } = xt(t), i = n.call(t, e);
			i ||= (e = /* @__PURE__ */ P(e), n.call(t, e));
			let a = r ? r.call(t, e) : void 0, o = t.delete(e);
			return i && et(t, "delete", e, void 0, a), o;
		},
		clear() {
			let e = /* @__PURE__ */ P(this), t = e.size !== 0, n = e.clear();
			return t && et(e, "clear", void 0, void 0, void 0), n;
		}
	}), [
		"keys",
		"values",
		"entries",
		Symbol.iterator
	].forEach((r) => {
		n[r] = St(r, e, t);
	}), n;
}
function Tt(e, t) {
	let n = wt(e, t);
	return (t, r, i) => r === "__v_isReactive" ? !e : r === "__v_isReadonly" ? e : r === "__v_raw" ? t : Reflect.get(u(n, r) && r in t ? n : t, r, i);
}
var Et = { get: /* @__PURE__ */ Tt(!1, !1) }, Dt = { get: /* @__PURE__ */ Tt(!1, !0) }, Ot = { get: /* @__PURE__ */ Tt(!0, !1) }, kt = /* @__PURE__ */ new WeakMap(), At = /* @__PURE__ */ new WeakMap(), jt = /* @__PURE__ */ new WeakMap(), Mt = /* @__PURE__ */ new WeakMap();
function Nt(e) {
	switch (e) {
		case "Object":
		case "Array": return 1;
		case "Map":
		case "Set":
		case "WeakMap":
		case "WeakSet": return 2;
		default: return 0;
	}
}
// @__NO_SIDE_EFFECTS__
function Pt(e) {
	return /* @__PURE__ */ zt(e) ? e : Lt(e, !1, _t, Et, kt);
}
// @__NO_SIDE_EFFECTS__
function Ft(e) {
	return Lt(e, !1, yt, Dt, At);
}
// @__NO_SIDE_EFFECTS__
function It(e) {
	return Lt(e, !0, vt, Ot, jt);
}
function Lt(e, t, n, r, i) {
	if (!v(e) || e.__v_raw && !(t && e.__v_isReactive) || e.__v_skip || !Object.isExtensible(e)) return e;
	let a = i.get(e);
	if (a) return a;
	let o = Nt(S(e));
	if (o === 0) return e;
	let s = new Proxy(e, o === 2 ? r : n);
	return i.set(e, s), s;
}
// @__NO_SIDE_EFFECTS__
function Rt(e) {
	return /* @__PURE__ */ zt(e) ? /* @__PURE__ */ Rt(e.__v_raw) : !!(e && e.__v_isReactive);
}
// @__NO_SIDE_EFFECTS__
function zt(e) {
	return !!(e && e.__v_isReadonly);
}
// @__NO_SIDE_EFFECTS__
function Bt(e) {
	return !!(e && e.__v_isShallow);
}
// @__NO_SIDE_EFFECTS__
function Vt(e) {
	return e ? !!e.__v_raw : !1;
}
// @__NO_SIDE_EFFECTS__
function P(e) {
	let t = e && e.__v_raw;
	return t ? /* @__PURE__ */ P(t) : e;
}
function Ht(e) {
	return !u(e, "__v_skip") && Object.isExtensible(e) && O(e, "__v_skip", !0), e;
}
var Ut = (e) => v(e) ? /* @__PURE__ */ Pt(e) : e, Wt = (e) => v(e) ? /* @__PURE__ */ It(e) : e;
// @__NO_SIDE_EFFECTS__
function F(e) {
	return e ? e.__v_isRef === !0 : !1;
}
// @__NO_SIDE_EFFECTS__
function Gt(e) {
	return Kt(e, !1);
}
function Kt(e, t) {
	return /* @__PURE__ */ F(e) ? e : new qt(e, t);
}
var qt = class {
	constructor(e, t) {
		this.dep = new Je(), this.__v_isRef = !0, this.__v_isShallow = !1, this._rawValue = t ? e : /* @__PURE__ */ P(e), this._value = t ? e : Ut(e), this.__v_isShallow = t;
	}
	get value() {
		return this.dep.track(), this._value;
	}
	set value(e) {
		let t = this._rawValue, n = this.__v_isShallow || /* @__PURE__ */ Bt(e) || /* @__PURE__ */ zt(e);
		e = n ? e : /* @__PURE__ */ P(e), ae(e, t) && (this._rawValue = e, this._value = n ? e : Ut(e), this.dep.trigger());
	}
};
function I(e) {
	return /* @__PURE__ */ F(e) ? e.value : e;
}
var Jt = {
	get: (e, t, n) => t === "__v_raw" ? e : I(Reflect.get(e, t, n)),
	set: (e, t, n, r) => {
		let i = e[t];
		return /* @__PURE__ */ F(i) && !/* @__PURE__ */ F(n) ? (i.value = n, !0) : Reflect.set(e, t, n, r);
	}
};
function Yt(e) {
	return /* @__PURE__ */ Rt(e) ? e : new Proxy(e, Jt);
}
var Xt = class {
	constructor(e, t, n) {
		this.fn = e, this.setter = t, this._value = void 0, this.dep = new Je(this), this.__v_isRef = !0, this.deps = void 0, this.depsTail = void 0, this.flags = 16, this.globalVersion = Ke - 1, this.next = void 0, this.effect = this, this.__v_isReadonly = !t, this.isSSR = n;
	}
	notify() {
		if (this.flags |= 16, !(this.flags & 8) && M !== this) return Me(this, !0), !0;
	}
	get value() {
		let e = this.dep.track();
		return Re(this), e && (e.version = this.dep.version), this._value;
	}
	set value(e) {
		this.setter && this.setter(e);
	}
};
// @__NO_SIDE_EFFECTS__
function Zt(e, t, n = !1) {
	let r, i;
	return h(e) ? r = e : (r = e.get, i = e.set), new Xt(r, i, n);
}
var Qt = {}, $t = /* @__PURE__ */ new WeakMap(), en = void 0;
function tn(e, t = !1, n = en) {
	if (n) {
		let t = $t.get(n);
		t || $t.set(n, t = []), t.push(e);
	}
}
function nn(e, n, i = t) {
	let { immediate: a, deep: o, once: s, scheduler: l, augmentJob: u, call: f } = i, p = (e) => o ? e : /* @__PURE__ */ Bt(e) || o === !1 || o === 0 ? rn(e, 1) : rn(e), m, g, _, v, y = !1, b = !1;
	if (/* @__PURE__ */ F(e) ? (g = () => e.value, y = /* @__PURE__ */ Bt(e)) : /* @__PURE__ */ Rt(e) ? (g = () => p(e), y = !0) : d(e) ? (b = !0, y = e.some((e) => /* @__PURE__ */ Rt(e) || /* @__PURE__ */ Bt(e)), g = () => e.map((e) => {
		if (/* @__PURE__ */ F(e)) return e.value;
		if (/* @__PURE__ */ Rt(e)) return p(e);
		if (h(e)) return f ? f(e, 2) : e();
	})) : g = h(e) ? n ? f ? () => f(e, 2) : e : () => {
		if (_) {
			Ue();
			try {
				_();
			} finally {
				We();
			}
		}
		let t = en;
		en = m;
		try {
			return f ? f(e, 3, [v]) : e(v);
		} finally {
			en = t;
		}
	} : r, n && o) {
		let e = g, t = o === !0 ? Infinity : o;
		g = () => rn(e(), t);
	}
	let x = Ee(), S = () => {
		m.stop(), x && x.active && c(x.effects, m);
	};
	if (s && n) {
		let e = n;
		n = (...t) => {
			let n = e(...t);
			return S(), n;
		};
	}
	let C = b ? Array(e.length).fill(Qt) : Qt, w = (e) => {
		if (!(!(m.flags & 1) || !m.dirty && !e)) if (n) {
			let t = m.run();
			if (e || o || y || (b ? t.some((e, t) => ae(e, C[t])) : ae(t, C))) {
				_ && _();
				let e = en;
				en = m;
				try {
					let e = [
						t,
						C === Qt ? void 0 : b && C[0] === Qt ? [] : C,
						v
					];
					C = t, f ? f(n, 3, e) : n(...e);
				} finally {
					en = e;
				}
			}
		} else m.run();
	};
	return u && u(w), m = new Oe(g), m.scheduler = l ? () => l(w, !1) : w, v = (e) => tn(e, !1, m), _ = m.onStop = () => {
		let e = $t.get(m);
		if (e) {
			if (f) f(e, 4);
			else for (let t of e) t();
			$t.delete(m);
		}
	}, n ? a ? w(!0) : C = m.run() : l ? l(w.bind(null, !0), !0) : m.run(), S.pause = m.pause.bind(m), S.resume = m.resume.bind(m), S.stop = S, S;
}
function rn(e, t = Infinity, n) {
	if (t <= 0 || !v(e) || e.__v_skip || (n ||= /* @__PURE__ */ new Map(), (n.get(e) || 0) >= t)) return e;
	if (n.set(e, t), t--, /* @__PURE__ */ F(e)) rn(e.value, t, n);
	else if (d(e)) for (let r = 0; r < e.length; r++) rn(e[r], t, n);
	else if (p(e) || f(e)) e.forEach((e) => {
		rn(e, t, n);
	});
	else if (C(e)) {
		for (let r in e) rn(e[r], t, n);
		for (let r of Object.getOwnPropertySymbols(e)) Object.prototype.propertyIsEnumerable.call(e, r) && rn(e[r], t, n);
	}
	return e;
}
//#endregion
//#region node_modules/@vue/runtime-core/dist/runtime-core.esm-bundler.js
function an(e, t, n, r) {
	try {
		return r ? e(...r) : e();
	} catch (e) {
		sn(e, t, n);
	}
}
function on(e, t, n, r) {
	if (h(e)) {
		let i = an(e, t, n, r);
		return i && y(i) && i.catch((e) => {
			sn(e, t, n);
		}), i;
	}
	if (d(e)) {
		let i = [];
		for (let a = 0; a < e.length; a++) i.push(on(e[a], t, n, r));
		return i;
	}
}
function sn(e, n, r, i = !0) {
	let a = n ? n.vnode : null, { errorHandler: o, throwUnhandledErrorInProduction: s } = n && n.appContext.config || t;
	if (n) {
		let t = n.parent, i = n.proxy, a = `https://vuejs.org/error-reference/#runtime-${r}`;
		for (; t;) {
			let n = t.ec;
			if (n) {
				for (let t = 0; t < n.length; t++) if (n[t](e, i, a) === !1) return;
			}
			t = t.parent;
		}
		if (o) {
			Ue(), an(o, null, 10, [
				e,
				i,
				a
			]), We();
			return;
		}
	}
	cn(e, r, a, i, s);
}
function cn(e, t, n, r = !0, i = !1) {
	if (i) throw e;
	console.error(e);
}
var ln = [], un = -1, dn = [], fn = null, pn = 0, mn = /* @__PURE__ */ Promise.resolve(), hn = null;
function gn(e) {
	let t = hn || mn;
	return e ? t.then(this ? e.bind(this) : e) : t;
}
function _n(e) {
	let t = un + 1, n = ln.length;
	for (; t < n;) {
		let r = t + n >>> 1, i = ln[r], a = Cn(i);
		a < e || a === e && i.flags & 2 ? t = r + 1 : n = r;
	}
	return t;
}
function vn(e) {
	if (!(e.flags & 1)) {
		let t = Cn(e), n = ln[ln.length - 1];
		!n || !(e.flags & 2) && t >= Cn(n) ? ln.push(e) : ln.splice(_n(t), 0, e), e.flags |= 1, yn();
	}
}
function yn() {
	hn ||= mn.then(wn);
}
function bn(e) {
	d(e) ? dn.push(...e) : fn && e.id === -1 ? fn.splice(pn + 1, 0, e) : e.flags & 1 || (dn.push(e), e.flags |= 1), yn();
}
function xn(e, t, n = un + 1) {
	for (; n < ln.length; n++) {
		let t = ln[n];
		if (t && t.flags & 2) {
			if (e && t.id !== e.uid) continue;
			ln.splice(n, 1), n--, t.flags & 4 && (t.flags &= -2), t(), t.flags & 4 || (t.flags &= -2);
		}
	}
}
function Sn(e) {
	if (dn.length) {
		let e = [...new Set(dn)].sort((e, t) => Cn(e) - Cn(t));
		if (dn.length = 0, fn) {
			fn.push(...e);
			return;
		}
		for (fn = e, pn = 0; pn < fn.length; pn++) {
			let e = fn[pn];
			e.flags & 4 && (e.flags &= -2), e.flags & 8 || e(), e.flags &= -2;
		}
		fn = null, pn = 0;
	}
}
var Cn = (e) => e.id == null ? e.flags & 2 ? -1 : Infinity : e.id;
function wn(e) {
	try {
		for (un = 0; un < ln.length; un++) {
			let e = ln[un];
			e && !(e.flags & 8) && (e.flags & 4 && (e.flags &= -2), an(e, e.i, e.i ? 15 : 14), e.flags & 4 || (e.flags &= -2));
		}
	} finally {
		for (; un < ln.length; un++) {
			let e = ln[un];
			e && (e.flags &= -2);
		}
		un = -1, ln.length = 0, Sn(e), hn = null, (ln.length || dn.length) && wn(e);
	}
}
var Tn = null, En = null;
function Dn(e) {
	let t = Tn;
	return Tn = e, En = e && e.type.__scopeId || null, t;
}
function On(e, t = Tn, n) {
	if (!t || e._n) return e;
	let r = (...n) => {
		r._d && na(-1);
		let i = Dn(t), a = Qi.length, o;
		try {
			o = e(...n);
		} finally {
			for (let e = Qi.length; e > a; e--) ea();
			Dn(i), r._d && na(1);
		}
		return o;
	};
	return r._n = !0, r._c = !0, r._d = !0, r;
}
function L(e, n) {
	if (Tn === null) return e;
	let r = La(Tn), i = e.dirs ||= [];
	for (let e = 0; e < n.length; e++) {
		let [a, o, s, c = t] = n[e];
		a && (h(a) && (a = {
			mounted: a,
			updated: a
		}), a.deep && rn(o), i.push({
			dir: a,
			instance: r,
			value: o,
			oldValue: void 0,
			arg: s,
			modifiers: c
		}));
	}
	return e;
}
function kn(e, t, n, r) {
	let i = e.dirs, a = t && t.dirs;
	for (let o = 0; o < i.length; o++) {
		let s = i[o];
		a && (s.oldValue = a[o].value);
		let c = s.dir[r];
		c && (Ue(), on(c, n, 8, [
			e.el,
			s,
			e,
			t
		]), We());
	}
}
function An(e, t) {
	if (xa) {
		let n = xa.provides, r = xa.parent && xa.parent.provides;
		r === n && (n = xa.provides = Object.create(r)), n[e] = t;
	}
}
function jn(e, t, n = !1) {
	let r = Sa();
	if (r || oi) {
		let i = oi ? oi._context.provides : r ? r.parent == null || r.ce ? r.vnode.appContext && r.vnode.appContext.provides : r.parent.provides : void 0;
		if (i && e in i) return i[e];
		if (arguments.length > 1) return n && h(t) ? t.call(r && r.proxy) : t;
	}
}
var Mn = /* @__PURE__ */ Symbol.for("v-scx"), Nn = () => jn(Mn);
function Pn(e, t, n) {
	return Fn(e, t, n);
}
function Fn(e, n, i = t) {
	let { immediate: a, deep: o, flush: c, once: l } = i, u = s({}, i), d = n && a || !n && c !== "post", f;
	if (Oa) {
		if (c === "sync") {
			let e = Nn();
			f = e.__watcherHandles ||= [];
		} else if (!d) {
			let e = () => {};
			return e.stop = r, e.resume = r, e.pause = r, e;
		}
	}
	let p = xa;
	u.call = (e, t, n) => on(e, p, t, n);
	let m = !1;
	c === "post" ? u.scheduler = (e) => {
		B(e, p && p.suspense);
	} : c !== "sync" && (m = !0, u.scheduler = (e, t) => {
		t ? e() : vn(e);
	}), u.augmentJob = (e) => {
		n && (e.flags |= 4), m && (e.flags |= 2, p && (e.id = p.uid, e.i = p));
	};
	let h = nn(e, n, u);
	return Oa && (f ? f.push(h) : d && h()), h;
}
function In(e, t, n) {
	let r = this.proxy, i = g(e) ? e.includes(".") ? Ln(r, e) : () => r[e] : e.bind(r, r), a;
	h(t) ? a = t : (a = t.handler, n = t);
	let o = Ta(this), s = Fn(i, a.bind(r), n);
	return o(), s;
}
function Ln(e, t) {
	let n = t.split(".");
	return () => {
		let t = e;
		for (let e = 0; e < n.length && t; e++) t = t[n[e]];
		return t;
	};
}
var Rn = /* @__PURE__ */ new WeakMap(), zn = /* @__PURE__ */ Symbol("_vte"), Bn = (e) => e.__isTeleport, Vn = (e) => e && (e.disabled || e.disabled === ""), Hn = (e) => e && (e.defer || e.defer === ""), Un = (e) => typeof SVGElement < "u" && e instanceof SVGElement, Wn = (e) => typeof MathMLElement == "function" && e instanceof MathMLElement, Gn = (e, t) => {
	let n = e && e.to;
	return g(n) ? t ? t(n) : null : n;
}, Kn = {
	name: "Teleport",
	__isTeleport: !0,
	process(e, t, n, r, i, a, o, s, c, l) {
		let { mc: u, pc: d, pbc: f, o: { insert: p, querySelector: m, createText: h, createComment: g, parentNode: _ } } = l, v = Vn(t.props), { dynamicChildren: y } = t, b = (e, t, n) => {
			e.shapeFlag & 16 && u(e.children, t, n, i, a, o, s, c);
		}, x = (e = t) => {
			let n = Vn(e.props), r = e.target = Gn(e.props, m), a = Zn(r, e, h, p);
			r && (o !== "svg" && Un(r) ? o = "svg" : o !== "mathml" && Wn(r) && (o = "mathml"), i && i.isCE && (i.ce._teleportTargets || (i.ce._teleportTargets = /* @__PURE__ */ new Set())).add(r), n || (b(e, r, a), Xn(e, !1)));
		}, S = (e) => {
			let t = () => {
				if (Rn.get(e) === t) {
					if (Rn.delete(e), Vn(e.props)) {
						let t = _(e.el) || n;
						b(e, t, e.anchor), Xn(e, !0);
					}
					x(e);
				}
			};
			Rn.set(e, t), B(t, a);
		};
		if (e == null) {
			let e = t.el = h(""), i = t.anchor = h("");
			if (p(e, n, r), p(i, n, r), Hn(t.props) || a && a.pendingBranch) {
				S(t);
				return;
			}
			v && (b(t, n, i), Xn(t, !0)), x();
		} else {
			t.el = e.el;
			let r = t.anchor = e.anchor, u = Rn.get(e);
			if (u) {
				u.flags |= 8, Rn.delete(e), S(t);
				return;
			}
			t.targetStart = e.targetStart;
			let p = t.target = e.target, h = t.targetAnchor = e.targetAnchor, g = Vn(e.props), _ = g ? n : p, b = g ? r : h;
			if (o === "svg" || Un(p) ? o = "svg" : (o === "mathml" || Wn(p)) && (o = "mathml"), y ? (f(e.dynamicChildren, y, _, i, a, o, s), Hi(e, t, !0)) : c || d(e, t, _, b, i, a, o, s, !1), v) g ? t.props && e.props && t.props.to !== e.props.to && (t.props.to = e.props.to) : qn(t, n, r, l, 1);
			else if ((t.props && t.props.to) !== (e.props && e.props.to)) {
				let e = Gn(t.props, m);
				e && (t.target = e, qn(t, e, null, l, 0));
			} else g && qn(t, p, h, l, 1);
			Xn(t, v);
		}
	},
	remove(e, t, n, { um: r, o: { remove: i } }, a) {
		let { shapeFlag: o, children: s, anchor: c, targetStart: l, targetAnchor: u, target: d, props: f } = e, p = Vn(f), m = a || !p, h = Rn.get(e);
		if (h && (h.flags |= 8, Rn.delete(e)), d && (i(l), i(u)), a && i(c), !h && (p || d) && o & 16) for (let e = 0; e < s.length; e++) {
			let i = s[e];
			r(i, t, n, m, !!i.dynamicChildren);
		}
	},
	move: qn,
	hydrate: Jn
};
function qn(e, t, n, { o: { insert: r }, m: i }, a = 2) {
	a === 0 && r(e.targetAnchor, t, n);
	let { el: o, anchor: s, shapeFlag: c, children: l, props: u } = e, d = a === 2;
	if (d && r(o, t, n), !Rn.has(e) && (!d || Vn(u)) && c & 16) for (let e = 0; e < l.length; e++) i(l[e], t, n, 2);
	d && r(s, t, n);
}
function Jn(e, t, n, r, i, a, { o: { nextSibling: o, parentNode: s, querySelector: c, insert: l, createText: u } }, d) {
	function f(e, n) {
		let r = n;
		for (; r;) {
			if (r && r.nodeType === 8) {
				if (r.data === "teleport start anchor") t.targetStart = r;
				else if (r.data === "teleport anchor") {
					t.targetAnchor = r, e._lpa = t.targetAnchor && o(t.targetAnchor);
					break;
				}
			}
			r = o(r);
		}
	}
	function p(e, t) {
		t.anchor = d(o(e), t, s(e), n, r, i, a);
	}
	let m = t.target = Gn(t.props, c), h = Vn(t.props);
	if (m) {
		let c = m._lpa || m.firstChild;
		t.shapeFlag & 16 && (h ? (p(e, t), f(m, c), t.targetAnchor || Zn(m, t, u, l, s(e) === m ? e : null)) : (t.anchor = o(e), f(m, c), t.targetAnchor || Zn(m, t, u, l), d(c && o(c), t, m, n, r, i, a))), Xn(t, h);
	} else h && t.shapeFlag & 16 && (p(e, t), t.targetStart = e, t.targetAnchor = o(e));
	return t.anchor && o(t.anchor);
}
var Yn = Kn;
function Xn(e, t) {
	let n = e.ctx;
	if (n && n.ut) {
		let r, i;
		for (t ? (r = e.el, i = e.anchor) : (r = e.targetStart, i = e.targetAnchor); r && r !== i;) r.nodeType === 1 && r.setAttribute("data-v-owner", n.uid), r = r.nextSibling;
		n.ut();
	}
}
function Zn(e, t, n, r, i = null) {
	let a = t.targetStart = n(""), o = t.targetAnchor = n("");
	return a[zn] = o, e && (r(a, e, i), r(o, e, i)), o;
}
var Qn = /* @__PURE__ */ Symbol("_leaveCb"), $n = /* @__PURE__ */ Symbol("_enterCb");
function er() {
	let e = {
		isMounted: !1,
		isLeaving: !1,
		isUnmounting: !1,
		leavingVNodes: /* @__PURE__ */ new Map()
	};
	return Or(() => {
		e.isMounted = !0;
	}), jr(() => {
		e.isUnmounting = !0;
	}), e;
}
var tr = [Function, Array], nr = {
	mode: String,
	appear: Boolean,
	persisted: Boolean,
	onBeforeEnter: tr,
	onEnter: tr,
	onAfterEnter: tr,
	onEnterCancelled: tr,
	onBeforeLeave: tr,
	onLeave: tr,
	onAfterLeave: tr,
	onLeaveCancelled: tr,
	onBeforeAppear: tr,
	onAppear: tr,
	onAfterAppear: tr,
	onAppearCancelled: tr
}, rr = (e) => {
	let t = e.subTree;
	return t.component ? rr(t.component) : t;
}, ir = {
	name: "BaseTransition",
	props: nr,
	setup(e, { slots: t }) {
		let n = Sa(), r = er();
		return () => {
			let i = t.default && fr(t.default(), !0), a = i && i.length ? ar(i) : n.subTree ? q() : void 0;
			if (!a) return;
			let o = /* @__PURE__ */ P(e), { mode: s } = o;
			if (r.isLeaving) return lr(a);
			let c = ur(a);
			if (!c) return lr(a);
			let l = cr(c, o, r, n, (e) => l = e);
			c.type !== Xi && dr(c, l);
			let u = n.subTree && ur(n.subTree);
			if (u && u.type !== Xi && !oa(u, c) && rr(n).type !== Xi) {
				let e = cr(u, o, r, n);
				if (dr(u, e), s === "out-in" && c.type !== Xi) return r.isLeaving = !0, e.afterLeave = () => {
					r.isLeaving = !1, n.job.flags & 8 || n.update(), delete e.afterLeave, u = void 0;
				}, lr(a);
				s === "in-out" && c.type !== Xi ? e.delayLeave = (e, t, n) => {
					let i = sr(r, u);
					i[String(u.key)] = u, e[Qn] = () => {
						t(), e[Qn] = void 0, delete l.delayedLeave, u = void 0;
					}, l.delayedLeave = () => {
						n(), delete l.delayedLeave, u = void 0;
					};
				} : u = void 0;
			} else u &&= void 0;
			return a;
		};
	}
};
function ar(e) {
	let t = e[0];
	if (e.length > 1) {
		for (let n of e) if (n.type !== Xi) {
			t = n;
			break;
		}
	}
	return t;
}
var or = ir;
function sr(e, t) {
	let { leavingVNodes: n } = e, r = n.get(t.type);
	return r || (r = /* @__PURE__ */ Object.create(null), n.set(t.type, r)), r;
}
function cr(e, t, n, r, i) {
	let { appear: a, mode: o, persisted: s = !1, onBeforeEnter: c, onEnter: l, onAfterEnter: u, onEnterCancelled: f, onBeforeLeave: p, onLeave: m, onAfterLeave: h, onLeaveCancelled: g, onBeforeAppear: _, onAppear: v, onAfterAppear: y, onAppearCancelled: b } = t, x = String(e.key), S = sr(n, e), C = (e, t) => {
		e && on(e, r, 9, t);
	}, w = (e, t) => {
		let n = t[1];
		C(e, t), d(e) ? e.every((e) => e.length <= 1) && n() : e.length <= 1 && n();
	}, T = {
		mode: o,
		persisted: s,
		beforeEnter(t) {
			let r = c;
			if (!n.isMounted) if (a) r = _ || c;
			else return;
			t[Qn] && t[Qn](!0);
			let i = S[x];
			i && oa(e, i) && i.el[Qn] && i.el[Qn](), C(r, [t]);
		},
		enter(t) {
			if (S[x] === e) return;
			let r = l, i = u, o = f;
			if (!n.isMounted) if (a) r = v || l, i = y || u, o = b || f;
			else return;
			let s = !1;
			t[$n] = (e) => {
				s || (s = !0, C(e ? o : i, [t]), T.delayedLeave && T.delayedLeave(), t[$n] = void 0);
			};
			let c = t[$n].bind(null, !1);
			r ? w(r, [t, c]) : c();
		},
		leave(t, r) {
			let i = String(e.key);
			if (t[$n] && t[$n](!0), n.isUnmounting) return r();
			C(p, [t]);
			let a = !1;
			t[Qn] = (n) => {
				a || (a = !0, r(), C(n ? g : h, [t]), t[Qn] = void 0, S[i] === e && delete S[i]);
			};
			let o = t[Qn].bind(null, !1);
			S[i] = e, m ? w(m, [t, o]) : o();
		},
		clone(e) {
			let a = cr(e, t, n, r, i);
			return i && i(a), a;
		}
	};
	return T;
}
function lr(e) {
	if (br(e)) return e = da(e), e.children = null, e;
}
function ur(e) {
	if (!br(e)) return Bn(e.type) && e.children ? ar(e.children) : e;
	if (e.component) return e.component.subTree;
	let { shapeFlag: t, children: n } = e;
	if (n) {
		if (t & 16) return n[0];
		if (t & 32 && h(n.default)) return n.default();
	}
}
function dr(e, t) {
	e.shapeFlag & 6 && e.component ? (e.transition = t, dr(e.component.subTree, t)) : e.shapeFlag & 128 ? (e.ssContent.transition = t.clone(e.ssContent), e.ssFallback.transition = t.clone(e.ssFallback)) : e.transition = t;
}
function fr(e, t = !1, n) {
	let r = [], i = 0;
	for (let a = 0; a < e.length; a++) {
		let o = e[a], s = n == null ? o.key : String(n) + String(o.key == null ? a : o.key);
		o.type === V ? (o.patchFlag & 128 && i++, r = r.concat(fr(o.children, t, s))) : (t || o.type !== Xi) && r.push(s == null ? o : da(o, { key: s }));
	}
	if (i > 1) for (let e = 0; e < r.length; e++) r[e].patchFlag = -2;
	return r;
}
// @__NO_SIDE_EFFECTS__
function pr(e, t) {
	return h(e) ? /* @__PURE__ */ s({ name: e.name }, t, { setup: e }) : e;
}
function mr(e) {
	e.ids = [
		e.ids[0] + e.ids[2]++ + "-",
		0,
		0
	];
}
function hr(e, t) {
	let n;
	return !!((n = Object.getOwnPropertyDescriptor(e, t)) && !n.configurable);
}
var gr = /* @__PURE__ */ new WeakMap();
function _r(e, n, r, a, o = !1) {
	if (d(e)) {
		e.forEach((e, t) => _r(e, n && (d(n) ? n[t] : n), r, a, o));
		return;
	}
	if (yr(a) && !o) {
		a.shapeFlag & 512 && a.type.__asyncResolved && a.component.subTree.component && _r(e, n, r, a.component.subTree);
		return;
	}
	let s = a.shapeFlag & 4 ? La(a.component) : a.el, l = o ? null : s, { i: f, r: p } = e, m = n && n.r, _ = f.refs === t ? f.refs = {} : f.refs, v = f.setupState, y = /* @__PURE__ */ P(v), b = v === t ? i : (e) => !hr(_, e) && u(y, e), x = (e, t) => !(t && hr(_, t));
	if (m != null && m !== p) {
		if (vr(n), g(m)) _[m] = null, b(m) && (v[m] = null);
		else if (/* @__PURE__ */ F(m)) {
			let e = n;
			x(m, e.k) && (m.value = null), e.k && (_[e.k] = null);
		}
	}
	if (h(p)) an(p, f, 12, [l, _]);
	else {
		let t = g(p), n = /* @__PURE__ */ F(p);
		if (t || n) {
			let i = () => {
				if (e.f) {
					let n = t ? b(p) ? v[p] : _[p] : x(p) || !e.k ? p.value : _[e.k];
					if (o) d(n) && c(n, s);
					else if (d(n)) n.includes(s) || n.push(s);
					else if (t) _[p] = [s], b(p) && (v[p] = _[p]);
					else {
						let t = [s];
						x(p, e.k) && (p.value = t), e.k && (_[e.k] = t);
					}
				} else t ? (_[p] = l, b(p) && (v[p] = l)) : n && (x(p, e.k) && (p.value = l), e.k && (_[e.k] = l));
			};
			if (l) {
				let t = () => {
					i(), gr.delete(e);
				};
				t.id = -1, gr.set(e, t), B(t, r);
			} else vr(e), i();
		}
	}
}
function vr(e) {
	let t = gr.get(e);
	t && (t.flags |= 8, gr.delete(e));
}
ue().requestIdleCallback, ue().cancelIdleCallback;
var yr = (e) => !!e.type.__asyncLoader, br = (e) => e.type.__isKeepAlive;
function xr(e, t) {
	Cr(e, "a", t);
}
function Sr(e, t) {
	Cr(e, "da", t);
}
function Cr(e, t, n = xa) {
	let r = e.__wdc ||= () => {
		let t = n;
		for (; t;) {
			if (t.isDeactivated) return;
			t = t.parent;
		}
		return e();
	};
	if (Tr(t, r, n), n) {
		let e = n.parent;
		for (; e && e.parent;) br(e.parent.vnode) && wr(r, t, n, e), e = e.parent;
	}
}
function wr(e, t, n, r) {
	let i = Tr(t, e, r, !0);
	Mr(() => {
		c(r[t], i);
	}, n);
}
function Tr(e, t, n = xa, r = !1) {
	if (n) {
		let i = n[e] || (n[e] = []), a = t.__weh ||= (...r) => {
			Ue();
			let i = Ta(n), a = on(t, n, e, r);
			return i(), We(), a;
		};
		return r ? i.unshift(a) : i.push(a), a;
	}
}
var Er = (e) => (t, n = xa) => {
	(!Oa || e === "sp") && Tr(e, (...e) => t(...e), n);
}, Dr = Er("bm"), Or = Er("m"), kr = Er("bu"), Ar = Er("u"), jr = Er("bum"), Mr = Er("um"), Nr = Er("sp"), Pr = Er("rtg"), Fr = Er("rtc");
function Ir(e, t = xa) {
	Tr("ec", e, t);
}
var Lr = /* @__PURE__ */ Symbol.for("v-ndc");
function R(e, t, n, r) {
	let i, a = n && n[r], o = d(e);
	if (o || g(e)) {
		let n = o && /* @__PURE__ */ Rt(e), r = !1, s = !1;
		n && (r = !/* @__PURE__ */ Bt(e), s = /* @__PURE__ */ zt(e), e = nt(e)), i = Array(e.length);
		for (let n = 0, o = e.length; n < o; n++) i[n] = t(r ? s ? Wt(Ut(e[n])) : Ut(e[n]) : e[n], n, void 0, a && a[n]);
	} else if (typeof e == "number") {
		i = Array(e);
		for (let n = 0; n < e; n++) i[n] = t(n + 1, n, void 0, a && a[n]);
	} else if (v(e)) if (e[Symbol.iterator]) i = Array.from(e, (e, n) => t(e, n, void 0, a && a[n]));
	else {
		let n = Object.keys(e);
		i = Array(n.length);
		for (let r = 0, o = n.length; r < o; r++) {
			let o = n[r];
			i[r] = t(e[o], o, r, a && a[r]);
		}
	}
	else i = [];
	return n && (n[r] = i), i;
}
var Rr = (e) => e ? Da(e) ? La(e) : Rr(e.parent) : null, zr = /* @__PURE__ */ s(/* @__PURE__ */ Object.create(null), {
	$: (e) => e,
	$el: (e) => e.vnode.el,
	$data: (e) => e.data,
	$props: (e) => e.props,
	$attrs: (e) => e.attrs,
	$slots: (e) => e.slots,
	$refs: (e) => e.refs,
	$parent: (e) => Rr(e.parent),
	$root: (e) => Rr(e.root),
	$host: (e) => e.ce,
	$emit: (e) => e.emit,
	$options: (e) => Jr(e),
	$forceUpdate: (e) => e.f ||= () => {
		vn(e.update);
	},
	$nextTick: (e) => e.n ||= gn.bind(e.proxy),
	$watch: (e) => In.bind(e)
}), Br = (e, n) => e !== t && !e.__isScriptSetup && u(e, n), Vr = {
	get({ _: e }, n) {
		if (n === "__v_skip") return !0;
		let { ctx: r, setupState: i, data: a, props: o, accessCache: s, type: c, appContext: l } = e;
		if (n[0] !== "$") {
			let e = s[n];
			if (e !== void 0) switch (e) {
				case 1: return i[n];
				case 2: return a[n];
				case 4: return r[n];
				case 3: return o[n];
			}
			else if (Br(i, n)) return s[n] = 1, i[n];
			else if (a !== t && u(a, n)) return s[n] = 2, a[n];
			else if (u(o, n)) return s[n] = 3, o[n];
			else if (r !== t && u(r, n)) return s[n] = 4, r[n];
			else Ur && (s[n] = 0);
		}
		let d = zr[n], f, p;
		if (d) return n === "$attrs" && N(e.attrs, "get", ""), d(e);
		if ((f = c.__cssModules) && (f = f[n])) return f;
		if (r !== t && u(r, n)) return s[n] = 4, r[n];
		if (p = l.config.globalProperties, u(p, n)) return p[n];
	},
	set({ _: e }, n, r) {
		let { data: i, setupState: a, ctx: o } = e;
		return Br(a, n) ? (a[n] = r, !0) : i !== t && u(i, n) ? (i[n] = r, !0) : u(e.props, n) || n[0] === "$" && n.slice(1) in e ? !1 : (o[n] = r, !0);
	},
	has({ _: { data: e, setupState: n, accessCache: r, ctx: i, appContext: a, props: o, type: s } }, c) {
		let l;
		return !!(r[c] || e !== t && c[0] !== "$" && u(e, c) || Br(n, c) || u(o, c) || u(i, c) || u(zr, c) || u(a.config.globalProperties, c) || (l = s.__cssModules) && l[c]);
	},
	defineProperty(e, t, n) {
		return n.get == null ? u(n, "value") && this.set(e, t, n.value, null) : e._.accessCache[t] = 0, Reflect.defineProperty(e, t, n);
	}
};
function Hr(e) {
	return d(e) ? e.reduce((e, t) => (e[t] = null, e), {}) : e;
}
var Ur = !0;
function Wr(e) {
	let t = Jr(e), n = e.proxy, i = e.ctx;
	Ur = !1, t.beforeCreate && Kr(t.beforeCreate, e, "bc");
	let { data: a, computed: o, methods: s, watch: c, provide: l, inject: u, created: f, beforeMount: p, mounted: m, beforeUpdate: g, updated: _, activated: y, deactivated: b, beforeDestroy: x, beforeUnmount: S, destroyed: C, unmounted: w, render: T, renderTracked: ee, renderTriggered: te, errorCaptured: E, serverPrefetch: ne, expose: D, inheritAttrs: re, components: ie, directives: ae, filters: oe } = t;
	if (u && Gr(u, i, null), s) for (let e in s) {
		let t = s[e];
		h(t) && (i[e] = t.bind(n));
	}
	if (a) {
		let t = a.call(n, n);
		v(t) && (e.data = /* @__PURE__ */ Pt(t));
	}
	if (Ur = !0, o) for (let e in o) {
		let t = o[e], a = J({
			get: h(t) ? t.bind(n, n) : h(t.get) ? t.get.bind(n, n) : r,
			set: !h(t) && h(t.set) ? t.set.bind(n) : r
		});
		Object.defineProperty(i, e, {
			enumerable: !0,
			configurable: !0,
			get: () => a.value,
			set: (e) => a.value = e
		});
	}
	if (c) for (let e in c) qr(c[e], i, n, e);
	if (l) {
		let e = h(l) ? l.call(n) : l;
		Reflect.ownKeys(e).forEach((t) => {
			An(t, e[t]);
		});
	}
	f && Kr(f, e, "c");
	function O(e, t) {
		d(t) ? t.forEach((t) => e(t.bind(n))) : t && e(t.bind(n));
	}
	if (O(Dr, p), O(Or, m), O(kr, g), O(Ar, _), O(xr, y), O(Sr, b), O(Ir, E), O(Fr, ee), O(Pr, te), O(jr, S), O(Mr, w), O(Nr, ne), d(D)) if (D.length) {
		let t = e.exposed ||= {};
		D.forEach((e) => {
			Object.defineProperty(t, e, {
				get: () => n[e],
				set: (t) => n[e] = t,
				enumerable: !0
			});
		});
	} else e.exposed ||= {};
	T && e.render === r && (e.render = T), re != null && (e.inheritAttrs = re), ie && (e.components = ie), ae && (e.directives = ae), ne && mr(e);
}
function Gr(e, t, n = r) {
	d(e) && (e = $r(e));
	for (let n in e) {
		let r = e[n], i;
		i = v(r) ? "default" in r ? jn(r.from || n, r.default, !0) : jn(r.from || n) : jn(r), /* @__PURE__ */ F(i) ? Object.defineProperty(t, n, {
			enumerable: !0,
			configurable: !0,
			get: () => i.value,
			set: (e) => i.value = e
		}) : t[n] = i;
	}
}
function Kr(e, t, n) {
	on(d(e) ? e.map((e) => e.bind(t.proxy)) : e.bind(t.proxy), t, n);
}
function qr(e, t, n, r) {
	let i = r.includes(".") ? Ln(n, r) : () => n[r];
	if (g(e)) {
		let n = t[e];
		h(n) && Pn(i, n);
	} else if (h(e)) Pn(i, e.bind(n));
	else if (v(e)) if (d(e)) e.forEach((e) => qr(e, t, n, r));
	else {
		let r = h(e.handler) ? e.handler.bind(n) : t[e.handler];
		h(r) && Pn(i, r, e);
	}
}
function Jr(e) {
	let t = e.type, { mixins: n, extends: r } = t, { mixins: i, optionsCache: a, config: { optionMergeStrategies: o } } = e.appContext, s = a.get(t), c;
	return s ? c = s : !i.length && !n && !r ? c = t : (c = {}, i.length && i.forEach((e) => Yr(c, e, o, !0)), Yr(c, t, o)), v(t) && a.set(t, c), c;
}
function Yr(e, t, n, r = !1) {
	let { mixins: i, extends: a } = t;
	a && Yr(e, a, n, !0), i && i.forEach((t) => Yr(e, t, n, !0));
	for (let i in t) if (!(r && i === "expose")) {
		let r = Xr[i] || n && n[i];
		e[i] = r ? r(e[i], t[i]) : t[i];
	}
	return e;
}
var Xr = {
	data: Zr,
	props: ti,
	emits: ti,
	methods: ei,
	computed: ei,
	beforeCreate: z,
	created: z,
	beforeMount: z,
	mounted: z,
	beforeUpdate: z,
	updated: z,
	beforeDestroy: z,
	beforeUnmount: z,
	destroyed: z,
	unmounted: z,
	activated: z,
	deactivated: z,
	errorCaptured: z,
	serverPrefetch: z,
	components: ei,
	directives: ei,
	watch: ni,
	provide: Zr,
	inject: Qr
};
function Zr(e, t) {
	return t ? e ? function() {
		return s(h(e) ? e.call(this, this) : e, h(t) ? t.call(this, this) : t);
	} : t : e;
}
function Qr(e, t) {
	return ei($r(e), $r(t));
}
function $r(e) {
	if (d(e)) {
		let t = {};
		for (let n = 0; n < e.length; n++) t[e[n]] = e[n];
		return t;
	}
	return e;
}
function z(e, t) {
	return e ? [...new Set([].concat(e, t))] : t;
}
function ei(e, t) {
	return e ? s(/* @__PURE__ */ Object.create(null), e, t) : t;
}
function ti(e, t) {
	return e ? d(e) && d(t) ? [.../* @__PURE__ */ new Set([...e, ...t])] : s(/* @__PURE__ */ Object.create(null), Hr(e), Hr(t ?? {})) : t;
}
function ni(e, t) {
	if (!e) return t;
	if (!t) return e;
	let n = s(/* @__PURE__ */ Object.create(null), e);
	for (let r in t) n[r] = z(e[r], t[r]);
	return n;
}
function ri() {
	return {
		app: null,
		config: {
			isNativeTag: i,
			performance: !1,
			globalProperties: {},
			optionMergeStrategies: {},
			errorHandler: void 0,
			warnHandler: void 0,
			compilerOptions: {}
		},
		mixins: [],
		components: {},
		directives: {},
		provides: /* @__PURE__ */ Object.create(null),
		optionsCache: /* @__PURE__ */ new WeakMap(),
		propsCache: /* @__PURE__ */ new WeakMap(),
		emitsCache: /* @__PURE__ */ new WeakMap()
	};
}
var ii = 0;
function ai(e, t) {
	return function(n, r = null) {
		h(n) || (n = s({}, n)), r != null && !v(r) && (r = null);
		let i = ri(), a = /* @__PURE__ */ new WeakSet(), o = [], c = !1, l = i.app = {
			_uid: ii++,
			_component: n,
			_props: r,
			_container: null,
			_context: i,
			_instance: null,
			version: Ba,
			get config() {
				return i.config;
			},
			set config(e) {},
			use(e, ...t) {
				return a.has(e) || (e && h(e.install) ? (a.add(e), e.install(l, ...t)) : h(e) && (a.add(e), e(l, ...t))), l;
			},
			mixin(e) {
				return i.mixins.includes(e) || i.mixins.push(e), l;
			},
			component(e, t) {
				return t ? (i.components[e] = t, l) : i.components[e];
			},
			directive(e, t) {
				return t ? (i.directives[e] = t, l) : i.directives[e];
			},
			mount(a, o, s) {
				if (!c) {
					let u = l._ceVNode || G(n, r);
					return u.appContext = i, s === !0 ? s = "svg" : s === !1 && (s = void 0), o && t ? t(u, a) : e(u, a, s), c = !0, l._container = a, a.__vue_app__ = l, La(u.component);
				}
			},
			onUnmount(e) {
				o.push(e);
			},
			unmount() {
				c && (on(o, l._instance, 16), e(null, l._container), delete l._container.__vue_app__);
			},
			provide(e, t) {
				return i.provides[e] = t, l;
			},
			runWithContext(e) {
				let t = oi;
				oi = l;
				try {
					return e();
				} finally {
					oi = t;
				}
			}
		};
		return l;
	};
}
var oi = null, si = (e, t) => t === "modelValue" || t === "model-value" ? e.modelModifiers : e[`${t}Modifiers`] || e[`${E(t)}Modifiers`] || e[`${D(t)}Modifiers`];
function ci(e, n, ...r) {
	if (e.isUnmounted) return;
	let i = e.vnode.props || t, a = r, o = n.startsWith("update:"), s = o && si(i, n.slice(7));
	s && (s.trim && (a = r.map((e) => g(e) ? e.trim() : e)), s.number && (a = r.map(se)));
	let c, l = i[c = ie(n)] || i[c = ie(E(n))];
	!l && o && (l = i[c = ie(D(n))]), l && on(l, e, 6, a);
	let u = i[c + "Once"];
	if (u) {
		if (!e.emitted) e.emitted = {};
		else if (e.emitted[c]) return;
		e.emitted[c] = !0, on(u, e, 6, a);
	}
}
var li = /* @__PURE__ */ new WeakMap();
function ui(e, t, n = !1) {
	let r = n ? li : t.emitsCache, i = r.get(e);
	if (i !== void 0) return i;
	let a = e.emits, o = {}, c = !1;
	if (!h(e)) {
		let r = (e) => {
			let n = ui(e, t, !0);
			n && (c = !0, s(o, n));
		};
		!n && t.mixins.length && t.mixins.forEach(r), e.extends && r(e.extends), e.mixins && e.mixins.forEach(r);
	}
	return !a && !c ? (v(e) && r.set(e, null), null) : (d(a) ? a.forEach((e) => o[e] = null) : s(o, a), v(e) && r.set(e, o), o);
}
function di(e, t) {
	return !e || !a(t) ? !1 : (t = t.slice(2), t = t === "Once" ? t : t.replace(/Once$/, ""), u(e, t[0].toLowerCase() + t.slice(1)) || u(e, D(t)) || u(e, t));
}
function fi(e) {
	let { type: t, vnode: n, proxy: r, withProxy: i, propsOptions: [a], slots: s, attrs: c, emit: l, render: u, renderCache: d, props: f, data: p, setupState: m, ctx: h, inheritAttrs: g } = e, _ = Dn(e), v, y;
	try {
		if (n.shapeFlag & 4) {
			let e = i || r, t = e;
			v = pa(u.call(t, e, d, f, m, p, h)), y = c;
		} else {
			let e = t;
			v = pa(e.length > 1 ? e(f, {
				attrs: c,
				slots: s,
				emit: l
			}) : e(f, null)), y = t.props ? c : pi(c);
		}
	} catch (t) {
		Qi.length = 0, sn(t, e, 1), v = G(Xi);
	}
	let b = v;
	if (y && g !== !1) {
		let e = Object.keys(y), { shapeFlag: t } = b;
		e.length && t & 7 && (a && e.some(o) && (y = mi(y, a)), b = da(b, y, !1, !0));
	}
	return n.dirs && (b = da(b, null, !1, !0), b.dirs = b.dirs ? b.dirs.concat(n.dirs) : n.dirs), n.transition && dr(b, n.transition), v = b, Dn(_), v;
}
var pi = (e) => {
	let t;
	for (let n in e) (n === "class" || n === "style" || a(n)) && ((t ||= {})[n] = e[n]);
	return t;
}, mi = (e, t) => {
	let n = {};
	for (let r in e) (!o(r) || !(r.slice(9) in t)) && (n[r] = e[r]);
	return n;
};
function hi(e, t, n) {
	let { props: r, children: i, component: a } = e, { props: o, children: s, patchFlag: c } = t, l = a.emitsOptions;
	if (t.dirs || t.transition) return !0;
	if (n && c >= 0) {
		if (c & 1024) return !0;
		if (c & 16) return r ? gi(r, o, l) : !!o;
		if (c & 8) {
			let e = t.dynamicProps;
			for (let t = 0; t < e.length; t++) {
				let n = e[t];
				if (_i(o, r, n) && !di(l, n)) return !0;
			}
		}
	} else return (i || s) && (!s || !s.$stable) ? !0 : r === o ? !1 : r ? !o || gi(r, o, l) : !!o;
	return !1;
}
function gi(e, t, n) {
	let r = Object.keys(t);
	if (r.length !== Object.keys(e).length) return !0;
	for (let i = 0; i < r.length; i++) {
		let a = r[i];
		if (_i(t, e, a) && !di(n, a)) return !0;
	}
	return !1;
}
function _i(e, t, n) {
	let r = e[n], i = t[n];
	return n === "style" && v(r) && v(i) ? !be(r, i) : r !== i;
}
function vi({ vnode: e, parent: t, suspense: n }, r) {
	for (; t;) {
		let n = t.subTree;
		if (n.suspense && n.suspense.activeBranch === e && (n.suspense.vnode.el = n.el = r, e = n), n === e) (e = t.vnode).el = r, t = t.parent;
		else break;
	}
	n && n.activeBranch === e && (n.vnode.el = r);
}
var yi = {}, bi = () => Object.create(yi), xi = (e) => Object.getPrototypeOf(e) === yi;
function Si(e, t, n, r = !1) {
	let i = {}, a = bi();
	e.propsDefaults = /* @__PURE__ */ Object.create(null), wi(e, t, i, a);
	for (let t in e.propsOptions[0]) t in i || (i[t] = void 0);
	e.props = n ? r ? i : /* @__PURE__ */ Ft(i) : e.type.props ? i : a, e.attrs = a;
}
function Ci(e, t, n, r) {
	let { props: i, attrs: a, vnode: { patchFlag: o } } = e, s = /* @__PURE__ */ P(i), [c] = e.propsOptions, l = !1;
	if ((r || o > 0) && !(o & 16)) {
		if (o & 8) {
			let n = e.vnode.dynamicProps;
			for (let r = 0; r < n.length; r++) {
				let o = n[r];
				if (di(e.emitsOptions, o)) continue;
				let d = t[o];
				if (c) if (u(a, o)) d !== a[o] && (a[o] = d, l = !0);
				else {
					let t = E(o);
					i[t] = Ti(c, s, t, d, e, !1);
				}
				else d !== a[o] && (a[o] = d, l = !0);
			}
		}
	} else {
		wi(e, t, i, a) && (l = !0);
		let r;
		for (let a in s) (!t || !u(t, a) && ((r = D(a)) === a || !u(t, r))) && (c ? n && (n[a] !== void 0 || n[r] !== void 0) && (i[a] = Ti(c, s, a, void 0, e, !0)) : delete i[a]);
		if (a !== s) for (let e in a) (!t || !u(t, e)) && (delete a[e], l = !0);
	}
	l && et(e.attrs, "set", "");
}
function wi(e, n, r, i) {
	let [a, o] = e.propsOptions, s = !1, c;
	if (n) for (let t in n) {
		if (T(t)) continue;
		let l = n[t], d;
		a && u(a, d = E(t)) ? !o || !o.includes(d) ? r[d] = l : (c ||= {})[d] = l : di(e.emitsOptions, t) || (!(t in i) || l !== i[t]) && (i[t] = l, s = !0);
	}
	if (o) {
		let n = /* @__PURE__ */ P(r), i = c || t;
		for (let t = 0; t < o.length; t++) {
			let s = o[t];
			r[s] = Ti(a, n, s, i[s], e, !u(i, s));
		}
	}
	return s;
}
function Ti(e, t, n, r, i, a) {
	let o = e[n];
	if (o != null) {
		let e = u(o, "default");
		if (e && r === void 0) {
			let e = o.default;
			if (o.type !== Function && !o.skipFactory && h(e)) {
				let { propsDefaults: a } = i;
				if (n in a) r = a[n];
				else {
					let o = Ta(i);
					r = a[n] = e.call(null, t), o();
				}
			} else r = e;
			i.ce && i.ce._setProp(n, r);
		}
		o[0] && (a && !e ? r = !1 : o[1] && (r === "" || r === D(n)) && (r = !0));
	}
	return r;
}
var Ei = /* @__PURE__ */ new WeakMap();
function Di(e, r, i = !1) {
	let a = i ? Ei : r.propsCache, o = a.get(e);
	if (o) return o;
	let c = e.props, l = {}, f = [], p = !1;
	if (!h(e)) {
		let t = (e) => {
			p = !0;
			let [t, n] = Di(e, r, !0);
			s(l, t), n && f.push(...n);
		};
		!i && r.mixins.length && r.mixins.forEach(t), e.extends && t(e.extends), e.mixins && e.mixins.forEach(t);
	}
	if (!c && !p) return v(e) && a.set(e, n), n;
	if (d(c)) for (let e = 0; e < c.length; e++) {
		let n = E(c[e]);
		Oi(n) && (l[n] = t);
	}
	else if (c) for (let e in c) {
		let t = E(e);
		if (Oi(t)) {
			let n = c[e], r = l[t] = d(n) || h(n) ? { type: n } : s({}, n), i = r.type, a = !1, o = !0;
			if (d(i)) for (let e = 0; e < i.length; ++e) {
				let t = i[e], n = h(t) && t.name;
				if (n === "Boolean") {
					a = !0;
					break;
				}
				n === "String" && (o = !1);
			}
			else a = h(i) && i.name === "Boolean";
			r[0] = a, r[1] = o, (a || u(r, "default")) && f.push(t);
		}
	}
	let m = [l, f];
	return v(e) && a.set(e, m), m;
}
function Oi(e) {
	return e[0] !== "$" && !T(e);
}
var ki = (e) => e === "_" || e === "_ctx" || e === "$stable", Ai = (e) => d(e) ? e.map(pa) : [pa(e)], ji = (e, t, n) => {
	if (t._n) return t;
	let r = On((...e) => Ai(t(...e)), n);
	return r._c = !1, r;
}, Mi = (e, t, n) => {
	let r = e._ctx;
	for (let n in e) {
		if (ki(n)) continue;
		let i = e[n];
		if (h(i)) t[n] = ji(n, i, r);
		else if (i != null) {
			let e = Ai(i);
			t[n] = () => e;
		}
	}
}, Ni = (e, t) => {
	let n = Ai(t);
	e.slots.default = () => n;
}, Pi = (e, t, n) => {
	for (let r in t) (n || !ki(r)) && (e[r] = t[r]);
}, Fi = (e, t, n) => {
	let r = e.slots = bi();
	if (e.vnode.shapeFlag & 32) {
		let e = t._;
		e ? (Pi(r, t, n), n && O(r, "_", e, !0)) : Mi(t, r);
	} else t && Ni(e, t);
}, Ii = (e, n, r) => {
	let { vnode: i, slots: a } = e, o = !0, s = t;
	if (i.shapeFlag & 32) {
		let e = n._;
		e ? r && e === 1 ? o = !1 : Pi(a, n, r) : (o = !n.$stable, Mi(n, a)), s = n;
	} else n && (Ni(e, n), s = { default: 1 });
	if (o) for (let e in a) !ki(e) && s[e] == null && delete a[e];
}, B = Ji;
function Li(e) {
	return Ri(e);
}
function Ri(e, i) {
	let a = ue();
	a.__VUE__ = !0;
	let { insert: o, remove: s, patchProp: c, createElement: l, createText: u, createComment: d, setText: f, setElementText: p, parentNode: m, nextSibling: h, setScopeId: g = r, insertStaticContent: _ } = e, v = (e, t, n, r = null, i = null, a = null, o = void 0, s = null, c = !!t.dynamicChildren) => {
		if (e === t) return;
		e && !oa(e, t) && (r = ye(e), he(e, i, a, !0), e = null), t.patchFlag === -2 && (c = !1, t.dynamicChildren = null);
		let { type: l, ref: u, shapeFlag: d } = t;
		switch (l) {
			case Yi:
				y(e, t, n, r);
				break;
			case Xi:
				b(e, t, n, r);
				break;
			case Zi:
				e ?? x(t, n, r, o);
				break;
			case V:
				ie(e, t, n, r, i, a, o, s, c);
				break;
			default: d & 1 ? w(e, t, n, r, i, a, o, s, c) : d & 6 ? ae(e, t, n, r, i, a, o, s, c) : (d & 64 || d & 128) && l.process(e, t, n, r, i, a, o, s, c, Se);
		}
		u != null && i ? _r(u, e && e.ref, a, t || e, !t) : u == null && e && e.ref != null && _r(e.ref, null, a, e, !0);
	}, y = (e, t, n, r) => {
		if (e == null) o(t.el = u(t.children), n, r);
		else {
			let n = t.el = e.el;
			t.children !== e.children && f(n, t.children);
		}
	}, b = (e, t, n, r) => {
		e == null ? o(t.el = d(t.children || ""), n, r) : t.el = e.el;
	}, x = (e, t, n, r) => {
		[e.el, e.anchor] = _(e.children, t, n, r, e.el, e.anchor);
	}, S = ({ el: e, anchor: t }, n, r) => {
		let i;
		for (; e && e !== t;) i = h(e), o(e, n, r), e = i;
		o(t, n, r);
	}, C = ({ el: e, anchor: t }) => {
		let n;
		for (; e && e !== t;) n = h(e), s(e), e = n;
		s(t);
	}, w = (e, t, n, r, i, a, o, s, c) => {
		if (t.type === "svg" ? o = "svg" : t.type === "math" && (o = "mathml"), e == null) ee(t, n, r, i, a, o, s, c);
		else {
			let n = e.el && e.el._isVueCE ? e.el : null;
			try {
				n && n._beginPatch(), ne(e, t, i, a, o, s, c);
			} finally {
				n && n._endPatch();
			}
		}
	}, ee = (e, t, n, r, i, a, s, u) => {
		let d, f, { props: m, shapeFlag: h, transition: g, dirs: _ } = e;
		if (d = e.el = l(e.type, a, m && m.is, m), h & 8 ? p(d, e.children) : h & 16 && E(e.children, d, null, r, i, zi(e, a), s, u), _ && kn(e, null, r, "created"), te(d, e, e.scopeId, s, r), m) {
			for (let e in m) e !== "value" && !T(e) && c(d, e, null, m[e], a, r);
			"value" in m && c(d, "value", null, m.value, a), (f = m.onVnodeBeforeMount) && _a(f, r, e);
		}
		_ && kn(e, null, r, "beforeMount");
		let v = Vi(i, g);
		v && g.beforeEnter(d), o(d, t, n), ((f = m && m.onVnodeMounted) || v || _) && B(() => {
			try {
				f && _a(f, r, e), v && g.enter(d), _ && kn(e, null, r, "mounted");
			} finally {}
		}, i);
	}, te = (e, t, n, r, i) => {
		if (n && g(e, n), r) for (let t = 0; t < r.length; t++) g(e, r[t]);
		if (i) {
			let n = i.subTree;
			if (t === n || qi(n.type) && (n.ssContent === t || n.ssFallback === t)) {
				let t = i.vnode;
				te(e, t, t.scopeId, t.slotScopeIds, i.parent);
			}
		}
	}, E = (e, t, n, r, i, a, o, s, c = 0) => {
		for (let l = c; l < e.length; l++) {
			let c = e[l] = s ? ma(e[l]) : pa(e[l]);
			v(null, c, t, n, r, i, a, o, s);
		}
	}, ne = (e, n, r, i, a, o, s) => {
		let l = n.el = e.el, { patchFlag: u, dynamicChildren: d, dirs: f } = n;
		u |= e.patchFlag & 16;
		let m = e.props || t, h = n.props || t, g;
		if (r && Bi(r, !1), (g = h.onVnodeBeforeUpdate) && _a(g, r, n, e), f && kn(n, e, r, "beforeUpdate"), r && Bi(r, !0), d && (!e.dynamicChildren || e.dynamicChildren.length !== d.length) && (u = 0, s = !1, d = null), (m.innerHTML && h.innerHTML == null || m.textContent && h.textContent == null) && p(l, ""), d ? D(e.dynamicChildren, d, l, r, i, zi(n, a), o) : s || de(e, n, l, null, r, i, zi(n, a), o, !1), u > 0) {
			if (u & 16) re(l, m, h, r, a);
			else if (u & 2 && m.class !== h.class && c(l, "class", null, h.class, a), u & 4 && c(l, "style", m.style, h.style, a), u & 8) {
				let e = n.dynamicProps;
				for (let t = 0; t < e.length; t++) {
					let n = e[t], i = m[n], o = h[n];
					(o !== i || n === "value") && c(l, n, i, o, a, r);
				}
			}
			u & 1 && e.children !== n.children && p(l, n.children);
		} else !s && d == null && re(l, m, h, r, a);
		((g = h.onVnodeUpdated) || f) && B(() => {
			g && _a(g, r, n, e), f && kn(n, e, r, "updated");
		}, i);
	}, D = (e, t, n, r, i, a, o) => {
		for (let s = 0; s < t.length; s++) {
			let c = e[s], l = t[s], u = c.el && (c.type === V || !oa(c, l) || c.shapeFlag & 198) ? m(c.el) : n;
			v(c, l, u, null, r, i, a, o, !0);
		}
	}, re = (e, n, r, i, a) => {
		if (n !== r) {
			if (n !== t) for (let t in n) !T(t) && !(t in r) && c(e, t, n[t], null, a, i);
			for (let t in r) {
				if (T(t)) continue;
				let o = r[t], s = n[t];
				o !== s && t !== "value" && c(e, t, s, o, a, i);
			}
			"value" in r && c(e, "value", n.value, r.value, a);
		}
	}, ie = (e, t, n, r, i, a, s, c, l) => {
		let d = t.el = e ? e.el : u(""), f = t.anchor = e ? e.anchor : u(""), { patchFlag: p, dynamicChildren: m, slotScopeIds: h } = t;
		h && (c = c ? c.concat(h) : h), e == null ? (o(d, n, r), o(f, n, r), E(t.children || [], n, f, i, a, s, c, l)) : p > 0 && p & 64 && m && e.dynamicChildren && e.dynamicChildren.length === m.length ? (D(e.dynamicChildren, m, n, i, a, s, c), (t.key != null || i && t === i.subTree) && Hi(e, t, !0)) : de(e, t, n, f, i, a, s, c, l);
	}, ae = (e, t, n, r, i, a, o, s, c) => {
		t.slotScopeIds = s, e == null ? t.shapeFlag & 512 ? i.ctx.activate(t, n, r, o, c) : O(t, n, r, i, a, o, c) : se(e, t, c);
	}, O = (e, t, n, r, i, a, o) => {
		let s = e.component = ba(e, r, i);
		if (br(e) && (s.ctx.renderer = Se), ka(s, !1, o), s.asyncDep) {
			if (i && i.registerDep(s, ce, o), !e.el) {
				let r = s.subTree = G(Xi);
				b(null, r, t, n), e.placeholder = r.el;
			}
		} else ce(s, e, t, n, i, a, o);
	}, se = (e, t, n) => {
		let r = t.component = e.component;
		if (hi(e, t, n)) if (r.asyncDep && !r.asyncResolved) {
			le(r, t, n);
			return;
		} else r.next = t, r.update();
		else t.el = e.el, r.vnode = t;
	}, ce = (e, t, n, r, i, a, o) => {
		let s = () => {
			if (e.isMounted) {
				let { next: t, bu: n, u: r, parent: s, vnode: c } = e;
				{
					let n = Wi(e);
					if (n) {
						t && (t.el = c.el, le(e, t, o)), n.asyncDep.then(() => {
							B(() => {
								e.isUnmounted || l();
							}, i);
						});
						return;
					}
				}
				let u = t, d;
				Bi(e, !1), t ? (t.el = c.el, le(e, t, o)) : t = c, n && oe(n), (d = t.props && t.props.onVnodeBeforeUpdate) && _a(d, s, t, c), Bi(e, !0);
				let f = fi(e), p = e.subTree;
				e.subTree = f, v(p, f, m(p.el), ye(p), e, i, a), t.el = f.el, u === null && vi(e, f.el), r && B(r, i), (d = t.props && t.props.onVnodeUpdated) && B(() => _a(d, s, t, c), i);
			} else {
				let o, { el: s, props: c } = t, { bm: l, m: u, parent: d, root: f, type: p } = e, m = yr(t);
				if (Bi(e, !1), l && oe(l), !m && (o = c && c.onVnodeBeforeMount) && _a(o, d, t), Bi(e, !0), s && Ce) {
					let t = () => {
						e.subTree = fi(e), Ce(s, e.subTree, e, i, null);
					};
					m && p.__asyncHydrate ? p.__asyncHydrate(s, e, t) : t();
				} else {
					f.ce && f.ce._hasShadowRoot() && f.ce._injectChildStyle(p, e.parent ? e.parent.type : void 0);
					let o = e.subTree = fi(e);
					v(null, o, n, r, e, i, a), t.el = o.el;
				}
				if (u && B(u, i), !m && (o = c && c.onVnodeMounted)) {
					let e = t;
					B(() => _a(o, d, e), i);
				}
				(t.shapeFlag & 256 || d && yr(d.vnode) && d.vnode.shapeFlag & 256) && e.a && B(e.a, i), e.isMounted = !0, t = n = r = null;
			}
		};
		e.scope.on();
		let c = e.effect = new Oe(s);
		e.scope.off();
		let l = e.update = c.run.bind(c), u = e.job = c.runIfDirty.bind(c);
		u.i = e, u.id = e.uid, c.scheduler = () => vn(u), Bi(e, !0), l();
	}, le = (e, t, n) => {
		t.component = e;
		let r = e.vnode.props;
		e.vnode = t, e.next = null, Ci(e, t.props, r, n), Ii(e, t.children, n), Ue(), xn(e), We();
	}, de = (e, t, n, r, i, a, o, s, c = !1) => {
		let l = e && e.children, u = e ? e.shapeFlag : 0, d = t.children, { patchFlag: f, shapeFlag: m } = t;
		if (f > 0) {
			if (f & 128) {
				pe(l, d, n, r, i, a, o, s, c);
				return;
			}
			if (f & 256) {
				fe(l, d, n, r, i, a, o, s, c);
				return;
			}
		}
		m & 8 ? (u & 16 && ve(l, i, a), d !== l && p(n, d)) : u & 16 ? m & 16 ? pe(l, d, n, r, i, a, o, s, c) : ve(l, i, a, !0) : (u & 8 && p(n, ""), m & 16 && E(d, n, r, i, a, o, s, c));
	}, fe = (e, t, r, i, a, o, s, c, l) => {
		e ||= n, t ||= n;
		let u = e.length, d = t.length, f = Math.min(u, d), p;
		for (p = 0; p < f; p++) {
			let n = t[p] = l ? ma(t[p]) : pa(t[p]);
			v(e[p], n, r, null, a, o, s, c, l);
		}
		u > d ? ve(e, a, o, !0, !1, f) : E(t, r, i, a, o, s, c, l, f);
	}, pe = (e, t, r, i, a, o, s, c, l) => {
		let u = 0, d = t.length, f = e.length - 1, p = d - 1;
		for (; u <= f && u <= p;) {
			let n = e[u], i = t[u] = l ? ma(t[u]) : pa(t[u]);
			if (oa(n, i)) v(n, i, r, null, a, o, s, c, l);
			else break;
			u++;
		}
		for (; u <= f && u <= p;) {
			let n = e[f], i = t[p] = l ? ma(t[p]) : pa(t[p]);
			if (oa(n, i)) v(n, i, r, null, a, o, s, c, l);
			else break;
			f--, p--;
		}
		if (u > f) {
			if (u <= p) {
				let e = p + 1, n = e < d ? t[e].el : i;
				for (; u <= p;) v(null, t[u] = l ? ma(t[u]) : pa(t[u]), r, n, a, o, s, c, l), u++;
			}
		} else if (u > p) for (; u <= f;) he(e[u], a, o, !0), u++;
		else {
			let m = u, h = u, g = /* @__PURE__ */ new Map();
			for (u = h; u <= p; u++) {
				let e = t[u] = l ? ma(t[u]) : pa(t[u]);
				e.key != null && g.set(e.key, u);
			}
			let _, y = 0, b = p - h + 1, x = !1, S = 0, C = Array(b);
			for (u = 0; u < b; u++) C[u] = 0;
			for (u = m; u <= f; u++) {
				let n = e[u];
				if (y >= b) {
					he(n, a, o, !0);
					continue;
				}
				let i;
				if (n.key != null) i = g.get(n.key);
				else for (_ = h; _ <= p; _++) if (C[_ - h] === 0 && oa(n, t[_])) {
					i = _;
					break;
				}
				i === void 0 ? he(n, a, o, !0) : (C[i - h] = u + 1, i >= S ? S = i : x = !0, v(n, t[i], r, null, a, o, s, c, l), y++);
			}
			let w = x ? Ui(C) : n;
			for (_ = w.length - 1, u = b - 1; u >= 0; u--) {
				let e = h + u, n = t[e], f = t[e + 1], p = e + 1 < d ? f.el || Ki(f) : i;
				C[u] === 0 ? v(null, n, r, p, a, o, s, c, l) : x && (_ < 0 || u !== w[_] ? me(n, r, p, 2) : _--);
			}
		}
	}, me = (e, t, n, r, i = null) => {
		let { el: a, type: c, transition: l, children: u, shapeFlag: d } = e;
		if (d & 6) {
			me(e.component.subTree, t, n, r);
			return;
		}
		if (d & 128) {
			e.suspense.move(t, n, r);
			return;
		}
		if (d & 64) {
			c.move(e, t, n, Se);
			return;
		}
		if (c === V) {
			o(a, t, n);
			for (let e = 0; e < u.length; e++) me(u[e], t, n, r);
			o(e.anchor, t, n);
			return;
		}
		if (c === Zi) {
			S(e, t, n);
			return;
		}
		if (r !== 2 && d & 1 && l) if (r === 0) l.persisted && !a[Qn] ? o(a, t, n) : (l.beforeEnter(a), o(a, t, n), B(() => l.enter(a), i));
		else {
			let { leave: r, delayLeave: i, afterLeave: c } = l, u = () => {
				e.ctx.isUnmounted ? s(a) : o(a, t, n);
			}, d = () => {
				let e = a._isLeaving || !!a[Qn];
				a._isLeaving && a[Qn](!0), l.persisted && !e ? u() : r(a, () => {
					u(), c && c();
				});
			};
			i ? i(a, u, d) : d();
		}
		else o(a, t, n);
	}, he = (e, t, n, r = !1, i = !1) => {
		let { type: a, props: o, ref: s, children: c, dynamicChildren: l, shapeFlag: u, patchFlag: d, dirs: f, cacheIndex: p, memo: m } = e;
		if (d === -2 && (i = !1), s != null && (Ue(), _r(s, null, n, e, !0), We()), p != null && (t.renderCache[p] = void 0), u & 256) {
			t.ctx.deactivate(e);
			return;
		}
		let h = u & 1 && f, g = !yr(e), _;
		if (g && (_ = o && o.onVnodeBeforeUnmount) && _a(_, t, e), u & 6) _e(e.component, n, r);
		else {
			if (u & 128) {
				e.suspense.unmount(n, r);
				return;
			}
			h && kn(e, null, t, "beforeUnmount"), u & 64 ? e.type.remove(e, t, n, Se, r) : l && !l.hasOnce && (a !== V || d > 0 && d & 64) ? ve(l, t, n, !1, !0) : (a === V && d & 384 || !i && u & 16) && ve(c, t, n), r && k(e);
		}
		let v = m != null && p == null;
		(g && (_ = o && o.onVnodeUnmounted) || h || v) && B(() => {
			_ && _a(_, t, e), h && kn(e, null, t, "unmounted"), v && (e.el = null);
		}, n);
	}, k = (e) => {
		let { type: t, el: n, anchor: r, transition: i } = e;
		if (t === V) {
			ge(n, r);
			return;
		}
		if (t === Zi) {
			C(e);
			return;
		}
		let a = () => {
			s(n), i && !i.persisted && i.afterLeave && i.afterLeave();
		};
		if (e.shapeFlag & 1 && i && !i.persisted) {
			let { leave: t, delayLeave: r } = i, o = () => t(n, a);
			r ? r(e.el, a, o) : o();
		} else a();
	}, ge = (e, t) => {
		let n;
		for (; e !== t;) n = h(e), s(e), e = n;
		s(t);
	}, _e = (e, t, n) => {
		let { bum: r, scope: i, job: a, subTree: o, um: s, m: c, a: l } = e;
		Gi(c), Gi(l), r && oe(r), i.stop(), a && (a.flags |= 8, he(o, e, t, n)), s && B(s, t), B(() => {
			e.isUnmounted = !0;
		}, t);
	}, ve = (e, t, n, r = !1, i = !1, a = 0) => {
		for (let o = a; o < e.length; o++) he(e[o], t, n, r, i);
	}, ye = (e) => {
		if (e.shapeFlag & 6) return ye(e.component.subTree);
		if (e.shapeFlag & 128) return e.suspense.next();
		let t = h(e.anchor || e.el), n = t && t[zn];
		return n ? h(n) : t;
	}, be = !1, xe = (e, t, n) => {
		let r;
		e == null ? t._vnode && (he(t._vnode, null, null, !0), r = t._vnode.component) : v(t._vnode || null, e, t, null, null, null, n), t._vnode = e, be ||= (be = !0, xn(r), Sn(), !1);
	}, Se = {
		p: v,
		um: he,
		m: me,
		r: k,
		mt: O,
		mc: E,
		pc: de,
		pbc: D,
		n: ye,
		o: e
	}, A, Ce;
	return i && ([A, Ce] = i(Se)), {
		render: xe,
		hydrate: A,
		createApp: ai(xe, A)
	};
}
function zi({ type: e, props: t }, n) {
	return n === "svg" && e === "foreignObject" || n === "mathml" && e === "annotation-xml" && t && t.encoding && t.encoding.includes("html") ? void 0 : n;
}
function Bi({ effect: e, job: t }, n) {
	n ? (e.flags |= 32, t.flags |= 4) : (e.flags &= -33, t.flags &= -5);
}
function Vi(e, t) {
	return (!e || e && !e.pendingBranch) && t && !t.persisted;
}
function Hi(e, t, n = !1) {
	let r = e.children, i = t.children;
	if (d(r) && d(i)) for (let e = 0; e < r.length; e++) {
		let t = r[e], a = i[e];
		a.shapeFlag & 1 && !a.dynamicChildren && ((a.patchFlag <= 0 || a.patchFlag === 32) && (a = i[e] = ma(i[e]), a.el = t.el), !n && a.patchFlag !== -2 && Hi(t, a)), a.type === Yi && (a.patchFlag === -1 && (a = i[e] = ma(a)), a.el = t.el), a.type === Xi && !a.el && (a.el = t.el);
	}
}
function Ui(e) {
	let t = e.slice(), n = [0], r, i, a, o, s, c = e.length;
	for (r = 0; r < c; r++) {
		let c = e[r];
		if (c !== 0) {
			if (i = n[n.length - 1], e[i] < c) {
				t[r] = i, n.push(r);
				continue;
			}
			for (a = 0, o = n.length - 1; a < o;) s = a + o >> 1, e[n[s]] < c ? a = s + 1 : o = s;
			c < e[n[a]] && (a > 0 && (t[r] = n[a - 1]), n[a] = r);
		}
	}
	for (a = n.length, o = n[a - 1]; a-- > 0;) n[a] = o, o = t[o];
	return n;
}
function Wi(e) {
	let t = e.subTree.component;
	if (t) return t.asyncDep && !t.asyncResolved ? t : Wi(t);
}
function Gi(e) {
	if (e) for (let t = 0; t < e.length; t++) e[t].flags |= 8;
}
function Ki(e) {
	if (e.placeholder) return e.placeholder;
	let t = e.component;
	return t ? Ki(t.subTree) : null;
}
var qi = (e) => e.__isSuspense;
function Ji(e, t) {
	t && t.pendingBranch ? d(e) ? t.effects.push(...e) : t.effects.push(e) : bn(e);
}
var V = /* @__PURE__ */ Symbol.for("v-fgt"), Yi = /* @__PURE__ */ Symbol.for("v-txt"), Xi = /* @__PURE__ */ Symbol.for("v-cmt"), Zi = /* @__PURE__ */ Symbol.for("v-stc"), Qi = [], $i = null;
function H(e = !1) {
	Qi.push($i = e ? null : []);
}
function ea() {
	Qi.pop(), $i = Qi[Qi.length - 1] || null;
}
var ta = 1;
function na(e, t = !1) {
	ta += e, e < 0 && $i && t && ($i.hasOnce = !0);
}
function ra(e) {
	return e.dynamicChildren = ta > 0 ? $i || n : null, ea(), ta > 0 && $i && $i.push(e), e;
}
function U(e, t, n, r, i, a) {
	return ra(W(e, t, n, r, i, a, !0));
}
function ia(e, t, n, r, i) {
	return ra(G(e, t, n, r, i, !0));
}
function aa(e) {
	return e ? e.__v_isVNode === !0 : !1;
}
function oa(e, t) {
	return e.type === t.type && e.key === t.key;
}
var sa = ({ key: e }) => e ?? null, ca = ({ ref: e, ref_key: t, ref_for: n }) => (typeof e == "number" && (e = "" + e), e == null ? null : g(e) || /* @__PURE__ */ F(e) || h(e) ? {
	i: Tn,
	r: e,
	k: t,
	f: !!n
} : e);
function W(e, t = null, n = null, r = 0, i = null, a = e === V ? 0 : 1, o = !1, s = !1) {
	let c = {
		__v_isVNode: !0,
		__v_skip: !0,
		type: e,
		props: t,
		key: t && sa(t),
		ref: t && ca(t),
		scopeId: En,
		slotScopeIds: null,
		children: n,
		component: null,
		suspense: null,
		ssContent: null,
		ssFallback: null,
		dirs: null,
		transition: null,
		el: null,
		anchor: null,
		target: null,
		targetStart: null,
		targetAnchor: null,
		staticCount: 0,
		shapeFlag: a,
		patchFlag: r,
		dynamicProps: i,
		dynamicChildren: null,
		appContext: null,
		ctx: Tn
	};
	return s ? (ha(c, n), a & 128 && e.normalize(c)) : n && (c.shapeFlag |= g(n) ? 8 : 16), ta > 0 && !o && $i && (c.patchFlag > 0 || a & 6) && c.patchFlag !== 32 && $i.push(c), c;
}
var G = la;
function la(e, t = null, n = null, r = 0, i = null, a = !1) {
	if ((!e || e === Lr) && (e = Xi), aa(e)) {
		let r = da(e, t, !0);
		return n && ha(r, n), ta > 0 && !a && $i && (r.shapeFlag & 6 ? $i[$i.indexOf(e)] = r : $i.push(r)), r.patchFlag = -2, r;
	}
	if (Ra(e) && (e = e.__vccOpts), t) {
		t = ua(t);
		let { class: e, style: n } = t;
		e && !g(e) && (t.class = k(e)), v(n) && (/* @__PURE__ */ Vt(n) && !d(n) && (n = s({}, n)), t.style = de(n));
	}
	let o = g(e) ? 1 : qi(e) ? 128 : Bn(e) ? 64 : v(e) ? 4 : h(e) ? 2 : 0;
	return W(e, t, n, r, i, o, a, !0);
}
function ua(e) {
	return e ? /* @__PURE__ */ Vt(e) || xi(e) ? s({}, e) : e : null;
}
function da(e, t, n = !1, r = !1) {
	let { props: i, ref: a, patchFlag: o, children: s, transition: c } = e, l = t ? ga(i || {}, t) : i, u = {
		__v_isVNode: !0,
		__v_skip: !0,
		type: e.type,
		props: l,
		key: l && sa(l),
		ref: t && t.ref ? n && a ? d(a) ? a.concat(ca(t)) : [a, ca(t)] : ca(t) : a,
		scopeId: e.scopeId,
		slotScopeIds: e.slotScopeIds,
		children: s,
		target: e.target,
		targetStart: e.targetStart,
		targetAnchor: e.targetAnchor,
		staticCount: e.staticCount,
		shapeFlag: e.shapeFlag,
		patchFlag: t && e.type !== V ? o === -1 ? 16 : o | 16 : o,
		dynamicProps: e.dynamicProps,
		dynamicChildren: e.dynamicChildren,
		appContext: e.appContext,
		dirs: e.dirs,
		transition: c,
		component: e.component,
		suspense: e.suspense,
		ssContent: e.ssContent && da(e.ssContent),
		ssFallback: e.ssFallback && da(e.ssFallback),
		placeholder: e.placeholder,
		el: e.el,
		anchor: e.anchor,
		ctx: e.ctx,
		ce: e.ce
	};
	return c && r && dr(u, c.clone(u)), u;
}
function K(e = " ", t = 0) {
	return G(Yi, null, e, t);
}
function fa(e, t) {
	let n = G(Zi, null, e);
	return n.staticCount = t, n;
}
function q(e = "", t = !1) {
	return t ? (H(), ia(Xi, null, e)) : G(Xi, null, e);
}
function pa(e) {
	return e == null || typeof e == "boolean" ? G(Xi) : d(e) ? G(V, null, e.slice()) : aa(e) ? ma(e) : G(Yi, null, String(e));
}
function ma(e) {
	return e.el === null && e.patchFlag !== -1 || e.memo ? e : da(e);
}
function ha(e, t) {
	let n = 0, { shapeFlag: r } = e;
	if (t == null) t = null;
	else if (d(t)) n = 16;
	else if (typeof t == "object") if (r & 65) {
		let n = t.default;
		n && (n._c && (n._d = !1), ha(e, n()), n._c && (n._d = !0));
		return;
	} else {
		n = 32;
		let r = t._;
		!r && !xi(t) ? t._ctx = Tn : r === 3 && Tn && (Tn.slots._ === 1 ? t._ = 1 : (t._ = 2, e.patchFlag |= 1024));
	}
	else if (h(t)) {
		if (r & 65) {
			ha(e, { default: t });
			return;
		}
		t = {
			default: t,
			_ctx: Tn
		}, n = 32;
	} else t = String(t), r & 64 ? (n = 16, t = [K(t)]) : n = 8;
	e.children = t, e.shapeFlag |= n;
}
function ga(...e) {
	let t = {};
	for (let n = 0; n < e.length; n++) {
		let r = e[n];
		for (let e in r) if (e === "class") t.class !== r.class && (t.class = k([t.class, r.class]));
		else if (e === "style") t.style = de([t.style, r.style]);
		else if (a(e)) {
			let n = t[e], i = r[e];
			i && n !== i && !(d(n) && n.includes(i)) ? t[e] = n ? [].concat(n, i) : i : i == null && n == null && !o(e) && (t[e] = i);
		} else e !== "" && (t[e] = r[e]);
	}
	return t;
}
function _a(e, t, n, r = null) {
	on(e, t, 7, [n, r]);
}
var va = ri(), ya = 0;
function ba(e, n, r) {
	let i = e.type, a = (n ? n.appContext : e.appContext) || va, o = {
		uid: ya++,
		vnode: e,
		type: i,
		parent: n,
		appContext: a,
		root: null,
		next: null,
		subTree: null,
		effect: null,
		update: null,
		job: null,
		scope: new Te(!0),
		render: null,
		proxy: null,
		exposed: null,
		exposeProxy: null,
		withProxy: null,
		provides: n ? n.provides : Object.create(a.provides),
		ids: n ? n.ids : [
			"",
			0,
			0
		],
		accessCache: null,
		renderCache: [],
		components: null,
		directives: null,
		propsOptions: Di(i, a),
		emitsOptions: ui(i, a),
		emit: null,
		emitted: null,
		propsDefaults: t,
		inheritAttrs: i.inheritAttrs,
		ctx: t,
		data: t,
		props: t,
		attrs: t,
		slots: t,
		refs: t,
		setupState: t,
		setupContext: null,
		suspense: r,
		suspenseId: r ? r.pendingId : 0,
		asyncDep: null,
		asyncResolved: !1,
		isMounted: !1,
		isUnmounted: !1,
		isDeactivated: !1,
		bc: null,
		c: null,
		bm: null,
		m: null,
		bu: null,
		u: null,
		um: null,
		bum: null,
		da: null,
		a: null,
		rtg: null,
		rtc: null,
		ec: null,
		sp: null
	};
	return o.ctx = { _: o }, o.root = n ? n.root : o, o.emit = ci.bind(null, o), e.ce && e.ce(o), o;
}
var xa = null, Sa = () => xa || Tn, Ca, wa;
{
	let e = ue(), t = (t, n) => {
		let r;
		return (r = e[t]) || (r = e[t] = []), r.push(n), (e) => {
			r.length > 1 ? r.forEach((t) => t(e)) : r[0](e);
		};
	};
	Ca = t("__VUE_INSTANCE_SETTERS__", (e) => xa = e), wa = t("__VUE_SSR_SETTERS__", (e) => Oa = e);
}
var Ta = (e) => {
	let t = xa;
	return Ca(e), e.scope.on(), () => {
		e.scope.off(), Ca(t);
	};
}, Ea = () => {
	xa && xa.scope.off(), Ca(null);
};
function Da(e) {
	return e.vnode.shapeFlag & 4;
}
var Oa = !1;
function ka(e, t = !1, n = !1) {
	t && wa(t);
	let { props: r, children: i } = e.vnode, a = Da(e);
	Si(e, r, a, t), Fi(e, i, n || t);
	let o = a ? Aa(e, t) : void 0;
	return t && wa(!1), o;
}
function Aa(e, t) {
	let n = e.type;
	e.accessCache = /* @__PURE__ */ Object.create(null), e.proxy = new Proxy(e.ctx, Vr);
	let { setup: r } = n;
	if (r) {
		Ue();
		let n = e.setupContext = r.length > 1 ? Ia(e) : null, i = Ta(e), a = an(r, e, 0, [e.props, n]), o = y(a);
		if (We(), i(), (o || e.sp) && !yr(e) && mr(e), o) {
			if (a.then(Ea, Ea), t) return a.then((n) => {
				ja(e, n, t);
			}).catch((t) => {
				sn(t, e, 0);
			});
			e.asyncDep = a;
		} else ja(e, a, t);
	} else Pa(e, t);
}
function ja(e, t, n) {
	h(t) ? e.type.__ssrInlineRender ? e.ssrRender = t : e.render = t : v(t) && (e.setupState = Yt(t)), Pa(e, n);
}
var Ma, Na;
function Pa(e, t, n) {
	let i = e.type;
	if (!e.render) {
		if (!t && Ma && !i.render) {
			let t = i.template || Jr(e).template;
			if (t) {
				let { isCustomElement: n, compilerOptions: r } = e.appContext.config, { delimiters: a, compilerOptions: o } = i;
				i.render = Ma(t, s(s({
					isCustomElement: n,
					delimiters: a
				}, r), o));
			}
		}
		e.render = i.render || r, Na && Na(e);
	}
	{
		let t = Ta(e);
		Ue();
		try {
			Wr(e);
		} finally {
			We(), t();
		}
	}
}
var Fa = { get(e, t) {
	return N(e, "get", ""), e[t];
} };
function Ia(e) {
	return {
		attrs: new Proxy(e.attrs, Fa),
		slots: e.slots,
		emit: e.emit,
		expose: (t) => {
			e.exposed = t || {};
		}
	};
}
function La(e) {
	return e.exposed ? e.exposeProxy ||= new Proxy(Yt(Ht(e.exposed)), {
		get(t, n) {
			if (n in t) return t[n];
			if (n in zr) return zr[n](e);
		},
		has(e, t) {
			return t in e || t in zr;
		}
	}) : e.proxy;
}
function Ra(e) {
	return h(e) && "__vccOpts" in e;
}
var J = (e, t) => /* @__PURE__ */ Zt(e, t, Oa);
function za(e, t, n) {
	try {
		na(-1);
		let r = arguments.length;
		return r === 2 ? v(t) && !d(t) ? aa(t) ? G(e, null, [t]) : G(e, t) : G(e, null, t) : (r > 3 ? n = Array.prototype.slice.call(arguments, 2) : r === 3 && aa(n) && (n = [n]), G(e, t, n));
	} finally {
		na(1);
	}
}
var Ba = "3.5.40", Va = void 0, Ha = typeof window < "u" && window.trustedTypes;
if (Ha) try {
	Va = /* @__PURE__ */ Ha.createPolicy("vue", { createHTML: (e) => e });
} catch {}
var Ua = Va ? (e) => Va.createHTML(e) : (e) => e, Wa = "http://www.w3.org/2000/svg", Ga = "http://www.w3.org/1998/Math/MathML", Ka = typeof document < "u" ? document : null, qa = Ka && /* @__PURE__ */ Ka.createElement("template"), Ja = {
	insert: (e, t, n) => {
		t.insertBefore(e, n || null);
	},
	remove: (e) => {
		let t = e.parentNode;
		t && t.removeChild(e);
	},
	createElement: (e, t, n, r) => {
		let i = t === "svg" ? Ka.createElementNS(Wa, e) : t === "mathml" ? Ka.createElementNS(Ga, e) : n ? Ka.createElement(e, { is: n }) : Ka.createElement(e);
		return e === "select" && r && r.multiple != null && i.setAttribute("multiple", r.multiple), i;
	},
	createText: (e) => Ka.createTextNode(e),
	createComment: (e) => Ka.createComment(e),
	setText: (e, t) => {
		e.nodeValue = t;
	},
	setElementText: (e, t) => {
		e.textContent = t;
	},
	parentNode: (e) => e.parentNode,
	nextSibling: (e) => e.nextSibling,
	querySelector: (e) => Ka.querySelector(e),
	setScopeId(e, t) {
		e.setAttribute(t, "");
	},
	insertStaticContent(e, t, n, r, i, a) {
		let o = n ? n.previousSibling : t.lastChild;
		if (i && (i === a || i.nextSibling)) for (; t.insertBefore(i.cloneNode(!0), n), !(i === a || !(i = i.nextSibling)););
		else {
			qa.innerHTML = Ua(r === "svg" ? `<svg>${e}</svg>` : r === "mathml" ? `<math>${e}</math>` : e);
			let i = qa.content;
			if (r === "svg" || r === "mathml") {
				let e = i.firstChild;
				for (; e.firstChild;) i.appendChild(e.firstChild);
				i.removeChild(e);
			}
			t.insertBefore(i, n);
		}
		return [o ? o.nextSibling : t.firstChild, n ? n.previousSibling : t.lastChild];
	}
}, Ya = "transition", Xa = "animation", Za = /* @__PURE__ */ Symbol("_vtc"), Qa = {
	name: String,
	type: String,
	css: {
		type: Boolean,
		default: !0
	},
	duration: [
		String,
		Number,
		Object
	],
	enterFromClass: String,
	enterActiveClass: String,
	enterToClass: String,
	appearFromClass: String,
	appearActiveClass: String,
	appearToClass: String,
	leaveFromClass: String,
	leaveActiveClass: String,
	leaveToClass: String
}, $a = /* @__PURE__ */ s({}, nr, Qa), eo = /* @__PURE__ */ ((e) => (e.displayName = "Transition", e.props = $a, e))((e, { slots: t }) => za(or, ro(e), t)), to = (e, t = []) => {
	d(e) ? e.forEach((e) => e(...t)) : e && e(...t);
}, no = (e) => e ? d(e) ? e.some((e) => e.length > 1) : e.length > 1 : !1;
function ro(e) {
	let t = {};
	for (let n in e) n in Qa || (t[n] = e[n]);
	if (e.css === !1) return t;
	let { name: n = "v", type: r, duration: i, enterFromClass: a = `${n}-enter-from`, enterActiveClass: o = `${n}-enter-active`, enterToClass: c = `${n}-enter-to`, appearFromClass: l = a, appearActiveClass: u = o, appearToClass: d = c, leaveFromClass: f = `${n}-leave-from`, leaveActiveClass: p = `${n}-leave-active`, leaveToClass: m = `${n}-leave-to` } = e, h = io(i), g = h && h[0], _ = h && h[1], { onBeforeEnter: v, onEnter: y, onEnterCancelled: b, onLeave: x, onLeaveCancelled: S, onBeforeAppear: C = v, onAppear: w = y, onAppearCancelled: T = b } = t, ee = (e, t, n, r) => {
		e._enterCancelled = r, so(e, t ? d : c), so(e, t ? u : o), n && n();
	}, te = (e, t) => {
		e._isLeaving = !1, so(e, f), so(e, m), so(e, p), t && t();
	}, E = (e) => (t, n) => {
		let i = e ? w : y, o = () => ee(t, e, n);
		to(i, [t, o]), co(() => {
			so(t, e ? l : a), oo(t, e ? d : c), no(i) || uo(t, r, g, o);
		});
	};
	return s(t, {
		onBeforeEnter(e) {
			to(v, [e]), oo(e, a), oo(e, o);
		},
		onBeforeAppear(e) {
			to(C, [e]), oo(e, l), oo(e, u);
		},
		onEnter: E(!1),
		onAppear: E(!0),
		onLeave(e, t) {
			e._isLeaving = !0;
			let n = () => te(e, t);
			oo(e, f), e._enterCancelled ? (oo(e, p), ho(e)) : (ho(e), oo(e, p)), co(() => {
				e._isLeaving && (so(e, f), oo(e, m), no(x) || uo(e, r, _, n));
			}), to(x, [e, n]);
		},
		onEnterCancelled(e) {
			ee(e, !1, void 0, !0), to(b, [e]);
		},
		onAppearCancelled(e) {
			ee(e, !0, void 0, !0), to(T, [e]);
		},
		onLeaveCancelled(e) {
			te(e), to(S, [e]);
		}
	});
}
function io(e) {
	if (e == null) return null;
	if (v(e)) return [ao(e.enter), ao(e.leave)];
	{
		let t = ao(e);
		return [t, t];
	}
}
function ao(e) {
	return ce(e);
}
function oo(e, t) {
	t.split(/\s+/).forEach((t) => t && e.classList.add(t)), (e[Za] || (e[Za] = /* @__PURE__ */ new Set())).add(t);
}
function so(e, t) {
	t.split(/\s+/).forEach((t) => t && e.classList.remove(t));
	let n = e[Za];
	n && (n.delete(t), n.size || (e[Za] = void 0));
}
function co(e) {
	requestAnimationFrame(() => {
		requestAnimationFrame(e);
	});
}
var lo = 0;
function uo(e, t, n, r) {
	let i = e._endId = ++lo, a = () => {
		i === e._endId && r();
	};
	if (n != null) return setTimeout(a, n);
	let { type: o, timeout: s, propCount: c } = fo(e, t);
	if (!o) return r();
	let l = o + "end", u = 0, d = () => {
		e.removeEventListener(l, f), a();
	}, f = (t) => {
		t.target === e && ++u >= c && d();
	};
	setTimeout(() => {
		u < c && d();
	}, s + 1), e.addEventListener(l, f);
}
function fo(e, t) {
	let n = window.getComputedStyle(e), r = (e) => (n[e] || "").split(", "), i = r(`${Ya}Delay`), a = r(`${Ya}Duration`), o = po(i, a), s = r(`${Xa}Delay`), c = r(`${Xa}Duration`), l = po(s, c), u = null, d = 0, f = 0;
	t === Ya ? o > 0 && (u = Ya, d = o, f = a.length) : t === Xa ? l > 0 && (u = Xa, d = l, f = c.length) : (d = Math.max(o, l), u = d > 0 ? o > l ? Ya : Xa : null, f = u ? u === Ya ? a.length : c.length : 0);
	let p = u === Ya && /\b(?:transform|all)(?:,|$)/.test(r(`${Ya}Property`).toString());
	return {
		type: u,
		timeout: d,
		propCount: f,
		hasTransform: p
	};
}
function po(e, t) {
	for (; e.length < t.length;) e = e.concat(e);
	return Math.max(...t.map((t, n) => mo(t) + mo(e[n])));
}
function mo(e) {
	return e === "auto" ? 0 : Number(e.slice(0, -1).replace(",", ".")) * 1e3;
}
function ho(e) {
	return (e ? e.ownerDocument : document).body.offsetHeight;
}
function go(e, t, n) {
	let r = e[Za];
	r && (t = (t ? [t, ...r] : [...r]).join(" ")), t == null ? e.removeAttribute("class") : n ? e.setAttribute("class", t) : e.className = t;
}
var _o = /* @__PURE__ */ Symbol("_vod"), vo = /* @__PURE__ */ Symbol("_vsh"), yo = /* @__PURE__ */ Symbol(""), bo = /(?:^|;)\s*display\s*:/;
function xo(e, t, n) {
	let r = e.style, i = g(n), a = !1;
	if (n && !i) {
		if (t) if (g(t)) for (let e of t.split(";")) {
			let t = e.slice(0, e.indexOf(":")).trim();
			n[t] ?? Co(r, t, "");
		}
		else for (let e in t) n[e] ?? Co(r, e, "");
		for (let i in n) {
			i === "display" && (a = !0);
			let o = n[i];
			o == null ? Co(r, i, "") : Do(e, i, !g(t) && t ? t[i] : void 0, o) || Co(r, i, o);
		}
	} else if (i) {
		if (t !== n) {
			let e = r[yo];
			e && (n += ";" + e), r.cssText = n, a = bo.test(n);
		}
	} else t && e.removeAttribute("style");
	_o in e && (e[_o] = a ? r.display : "", e[vo] && (r.display = "none"));
}
var So = /\s*!important$/;
function Co(e, t, n) {
	if (d(n)) n.forEach((n) => Co(e, t, n));
	else if (n ??= "", t.startsWith("--")) e.setProperty(t, n);
	else {
		let r = Eo(e, t);
		So.test(n) ? e.setProperty(D(r), n.replace(So, ""), "important") : e[r] = n;
	}
}
var wo = [
	"Webkit",
	"Moz",
	"ms"
], To = {};
function Eo(e, t) {
	let n = To[t];
	if (n) return n;
	let r = E(t);
	if (r !== "filter" && r in e) return To[t] = r;
	r = re(r);
	for (let n = 0; n < wo.length; n++) {
		let i = wo[n] + r;
		if (i in e) return To[t] = i;
	}
	return t;
}
function Do(e, t, n, r) {
	return e.tagName === "TEXTAREA" && (t === "width" || t === "height") && g(r) && n === r;
}
var Oo = "http://www.w3.org/1999/xlink";
function ko(e, t, n, r, i, a = _e(t)) {
	r && t.startsWith("xlink:") ? n == null ? e.removeAttributeNS(Oo, t.slice(6, t.length)) : e.setAttributeNS(Oo, t, n) : n == null || a && !ve(n) ? e.removeAttribute(t) : e.setAttribute(t, a ? "" : _(n) ? String(n) : n);
}
function Ao(e, t, n, r, i) {
	if (t === "innerHTML" || t === "textContent") {
		n != null && (e[t] = t === "innerHTML" ? Ua(n) : n);
		return;
	}
	let a = e.tagName;
	if (t === "value" && a !== "PROGRESS" && !a.includes("-")) {
		let r = a === "OPTION" ? e.getAttribute("value") || "" : e.value, i = n == null ? e.type === "checkbox" ? "on" : "" : String(n);
		(r !== i || !("_value" in e)) && (e.value = i), n ?? e.removeAttribute(t), e._value = n;
		return;
	}
	let o = !1;
	if (n === "" || n == null) {
		let r = typeof e[t];
		r === "boolean" ? n = ve(n) : n == null && r === "string" ? (n = "", o = !0) : r === "number" && (n = 0, o = !0);
	}
	try {
		e[t] = n;
	} catch {}
	o && e.removeAttribute(i || t);
}
function jo(e, t, n, r) {
	e.addEventListener(t, n, r);
}
function Mo(e, t, n, r) {
	e.removeEventListener(t, n, r);
}
var No = /* @__PURE__ */ Symbol("_vei");
function Po(e, t, n, r, i = null) {
	let a = e[No] || (e[No] = {}), o = a[t];
	if (r && o) o.value = r;
	else {
		let [n, s] = Lo(t);
		r ? jo(e, n, a[t] = Vo(r, i), s) : o && (Mo(e, n, o, s), a[t] = void 0);
	}
}
var Fo = /(Once|Passive|Capture)$/, Io = /^on:?(?:Once|Passive|Capture)$/;
function Lo(e) {
	let t, n;
	for (; (n = e.match(Fo)) && !Io.test(e);) t ||= {}, e = e.slice(0, e.length - n[1].length), t[n[1].toLowerCase()] = !0;
	return [e[2] === ":" ? e.slice(3) : D(e.slice(2)), t];
}
var Ro = 0, zo = /* @__PURE__ */ Promise.resolve(), Bo = () => Ro ||= (zo.then(() => Ro = 0), Date.now());
function Vo(e, t) {
	let n = (e) => {
		if (!e._vts) e._vts = Date.now();
		else if (e._vts <= n.attached) return;
		let r = n.value;
		if (d(r)) {
			let n = e.stopImmediatePropagation;
			e.stopImmediatePropagation = () => {
				n.call(e), e._stopped = !0;
			};
			let i = r.slice(), a = [e];
			for (let n = 0; n < i.length && !e._stopped; n++) {
				let e = i[n];
				e && on(e, t, 5, a);
			}
		} else on(r, t, 5, [e]);
	};
	return n.value = e, n.attached = Bo(), n;
}
var Ho = (e) => e.charCodeAt(0) === 111 && e.charCodeAt(1) === 110 && e.charCodeAt(2) > 96 && e.charCodeAt(2) < 123, Uo = (e, t, n, r, i, s) => {
	let c = i === "svg";
	t === "class" ? go(e, r, c) : t === "style" ? xo(e, n, r) : a(t) ? o(t) || Po(e, t, n, r, s) : (t[0] === "." ? (t = t.slice(1), !0) : t[0] === "^" ? (t = t.slice(1), !1) : Wo(e, t, r, c)) ? (Ao(e, t, r), !e.tagName.includes("-") && (t === "value" || t === "checked" || t === "selected") && ko(e, t, r, c, s, t !== "value")) : e._isVueCE && (Go(e, t) || e._def.__asyncLoader && (/[A-Z]/.test(t) || !g(r))) ? Ao(e, E(t), r, s, t) : (t === "true-value" ? e._trueValue = r : t === "false-value" && (e._falseValue = r), ko(e, t, r, c));
};
function Wo(e, t, n, r) {
	if (r) return !!(t === "innerHTML" || t === "textContent" || t in e && Ho(t) && h(n));
	if (t === "spellcheck" || t === "draggable" || t === "translate" || t === "autocorrect" || t === "sandbox" && e.tagName === "IFRAME" || t === "form" || t === "list" && e.tagName === "INPUT" || t === "type" && e.tagName === "TEXTAREA") return !1;
	if (t === "width" || t === "height") {
		let t = e.tagName;
		if (t === "IMG" || t === "VIDEO" || t === "CANVAS" || t === "SOURCE") return !1;
	}
	return Ho(t) && g(n) ? !1 : t in e;
}
function Go(e, t) {
	let n = e._def.props;
	if (!n) return !1;
	let r = E(t);
	return Array.isArray(n) ? n.some((e) => E(e) === r) : Object.keys(n).some((e) => E(e) === r);
}
var Ko = (e) => {
	let t = e.props["onUpdate:modelValue"] || !1;
	return d(t) ? (e) => oe(t, e) : t;
};
function qo(e) {
	e.target.composing = !0;
}
function Jo(e) {
	let t = e.target;
	t.composing && (t.composing = !1, t.dispatchEvent(new Event("input")));
}
var Yo = /* @__PURE__ */ Symbol("_assign");
function Xo(e, t, n) {
	return t && (e = e.trim()), n && (e = se(e)), e;
}
var Zo = {
	created(e, { modifiers: { lazy: t, trim: n, number: r } }, i) {
		e[Yo] = Ko(i);
		let a = r || i.props && i.props.type === "number";
		jo(e, t ? "change" : "input", (t) => {
			t.target.composing || e[Yo](Xo(e.value, n, a));
		}), (n || a) && jo(e, "change", () => {
			e.value = Xo(e.value, n, a);
		}), t || (jo(e, "compositionstart", qo), jo(e, "compositionend", Jo), jo(e, "change", Jo));
	},
	mounted(e, { value: t }) {
		e.value = t ?? "";
	},
	beforeUpdate(e, { value: t, oldValue: n, modifiers: { lazy: r, trim: i, number: a } }, o) {
		if (e[Yo] = Ko(o), e.composing) return;
		let s = (a || e.type === "number") && !/^0\d/.test(e.value) ? se(e.value) : e.value, c = t ?? "";
		if (s === c) return;
		let l = e.getRootNode();
		(l instanceof Document || l instanceof ShadowRoot) && l.activeElement === e && e.type !== "range" && (r && t === n || i && e.value.trim() === c) || (e.value = c);
	}
}, Qo = {
	deep: !0,
	created(e, t, n) {
		e[Yo] = Ko(n), jo(e, "change", () => {
			let t = e._modelValue, n = ns(e), r = e.checked, i = e[Yo];
			if (d(t)) {
				let e = xe(t, n), a = e !== -1;
				if (r && !a) i(t.concat(n));
				else if (!r && a) {
					let n = [...t];
					n.splice(e, 1), i(n);
				}
			} else if (p(t)) {
				let e = new Set(t);
				r ? e.add(n) : e.delete(n), i(e);
			} else i(rs(e, r));
		});
	},
	mounted: $o,
	beforeUpdate(e, t, n) {
		e[Yo] = Ko(n), $o(e, t, n);
	}
};
function $o(e, { value: t, oldValue: n }, r) {
	e._modelValue = t;
	let i;
	if (d(t)) i = xe(t, r.props.value) > -1;
	else if (p(t)) i = t.has(r.props.value);
	else {
		if (t === n) return;
		i = be(t, rs(e, !0));
	}
	e.checked !== i && (e.checked = i);
}
var es = {
	deep: !0,
	created(e, { value: t, modifiers: { number: n } }, r) {
		e._modelValue = t, jo(e, "change", () => {
			let t = Array.prototype.filter.call(e.options, (e) => e.selected).map((e) => n ? se(ns(e)) : ns(e));
			e[Yo](e.multiple ? p(e._modelValue) ? new Set(t) : t : t[0]), e._assigning = !0, gn(() => {
				e._assigning = !1;
			});
		}), e[Yo] = Ko(r);
	},
	mounted(e, { value: t }) {
		ts(e, t);
	},
	beforeUpdate(e, { value: t }, n) {
		e._modelValue = t, e[Yo] = Ko(n);
	},
	updated(e, { value: t }) {
		e._assigning || ts(e, t);
	}
};
function ts(e, t) {
	let n = e.multiple, r = d(t);
	if (!(n && !r && !p(t))) {
		for (let i = 0, a = e.options.length; i < a; i++) {
			let a = e.options[i], o = ns(a);
			if (n) if (r) {
				let e = typeof o;
				a.selected = e === "string" || e === "number" ? t.some((e) => String(e) === String(o)) : xe(t, o) > -1;
			} else a.selected = t.has(o);
			else if (be(ns(a), t)) {
				e.selectedIndex !== i && (e.selectedIndex = i);
				return;
			}
		}
		!n && e.selectedIndex !== -1 && (e.selectedIndex = -1);
	}
}
function ns(e) {
	return "_value" in e ? e._value : e.value;
}
function rs(e, t) {
	let n = t ? "_trueValue" : "_falseValue";
	return n in e ? e[n] : t;
}
var is = [
	"ctrl",
	"shift",
	"alt",
	"meta"
], as = {
	stop: (e) => e.stopPropagation(),
	prevent: (e) => e.preventDefault(),
	self: (e) => e.target !== e.currentTarget,
	ctrl: (e) => !e.ctrlKey,
	shift: (e) => !e.shiftKey,
	alt: (e) => !e.altKey,
	meta: (e) => !e.metaKey,
	left: (e) => "button" in e && e.button !== 0,
	middle: (e) => "button" in e && e.button !== 1,
	right: (e) => "button" in e && e.button !== 2,
	exact: (e, t) => is.some((n) => e[`${n}Key`] && !t.includes(n))
}, os = (e, t) => {
	if (!e) return e;
	let n = e._withMods ||= {}, r = t.join(".");
	return n[r] || (n[r] = ((n, ...r) => {
		for (let e = 0; e < t.length; e++) {
			let r = as[t[e]];
			if (r && r(n, t)) return;
		}
		return e(n, ...r);
	}));
}, ss = {
	esc: "escape",
	space: " ",
	up: "arrow-up",
	left: "arrow-left",
	right: "arrow-right",
	down: "arrow-down",
	delete: "backspace"
}, cs = (e, t) => {
	let n = e._withKeys ||= {}, r = t.join(".");
	return n[r] || (n[r] = ((n) => {
		if (!("key" in n)) return;
		let r = D(n.key);
		if (t.some((e) => e === r || ss[e] === r)) return e(n);
	}));
}, ls = /* @__PURE__ */ s({ patchProp: Uo }, Ja), us;
function ds() {
	return us ||= Li(ls);
}
var fs = ((...e) => {
	let t = ds().createApp(...e), { mount: n } = t;
	return t.mount = (e) => {
		let r = ms(e);
		if (!r) return;
		let i = t._component;
		!h(i) && !i.render && !i.template && (i.template = r.innerHTML), r.nodeType === 1 && (r.textContent = "");
		let a = n(r, !1, ps(r));
		return r instanceof Element && (r.removeAttribute("v-cloak"), r.setAttribute("data-v-app", "")), a;
	}, t;
});
function ps(e) {
	if (e instanceof SVGElement) return "svg";
	if (typeof MathMLElement == "function" && e instanceof MathMLElement) return "mathml";
}
function ms(e) {
	return g(e) ? document.querySelector(e) : e;
}
//#endregion
//#region src/api.ts
async function hs(e, t = {}) {
	let n = await fetch(e, {
		credentials: "same-origin",
		...t,
		headers: {
			Accept: "application/json",
			...t.headers || {}
		}
	}), r = (n.headers.get("content-type") || "").includes("application/json") ? await n.json() : await n.text();
	if (!n.ok) {
		let e = typeof r == "object" && r ? r.error || r.detail || r.message : r;
		throw Error(String(e || `Request failed (${n.status})`));
	}
	return r;
}
function gs(e) {
	return hs(e);
}
function _s(e, t = {}) {
	return hs(e, {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify(t)
	});
}
function vs(e, t) {
	return hs(e, {
		method: "PUT",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify(t)
	});
}
//#endregion
//#region src/trainerStore.ts
var ys = () => ({
	running: !1,
	exit_code: null,
	log_lines: [],
	progress: null
}), bs = () => ({
	personal: [],
	negative: [],
	personal_count: 0,
	negative_count: 0
}), xs = () => ({
	items: [],
	captured_count: 0,
	personal_count: 0,
	negative_count: 0
}), Ss = () => ({
	items: [],
	total_size_bytes: 0,
	total_file_count: 0
}), Cs = () => ({
	enabled: !1,
	wake_phrase: "",
	language: "en",
	english_accent: "mixed",
	stt_engine: "faster_whisper",
	minimum_transcript_chars: 2,
	delete_confirmed_wakes: !1,
	promote_close_misses: !1,
	schedule_hours: 24,
	minimum_new_negatives: 3,
	advertised_base_url: "",
	tater_url: "http://127.0.0.1:8501",
	notify_satellites: !0,
	train_microwakeword: !0,
	train_openwakeword: !1,
	train_nanowakeword_cnn: !1,
	train_nanowakeword_gru: !1,
	generated_negative_phrases: ""
}), Y = /* @__PURE__ */ Pt({
	activeView: "trainer",
	initialized: !1,
	busy: /* @__PURE__ */ new Set(),
	phrase: "",
	language: "en",
	englishAccent: "mixed",
	ttsMode: "hybrid",
	trainMicroWakeWord: !0,
	trainOpenWakeWord: !1,
	trainNanoWakeWordCnn: !1,
	trainNanoWakeWordGru: !1,
	generatedNegativePhrases: "",
	languages: [{
		code: "en",
		label: "English (en)",
		engines: ["omnivoice"]
	}],
	englishAccents: [
		{
			code: "mixed",
			label: "Mixed English"
		},
		{
			code: "australian",
			label: "Australian"
		},
		{
			code: "american",
			label: "American"
		},
		{
			code: "british",
			label: "British"
		},
		{
			code: "canadian",
			label: "Canadian"
		},
		{
			code: "irish",
			label: "Irish"
		},
		{
			code: "scottish",
			label: "Scottish"
		},
		{
			code: "new_zealand",
			label: "New Zealand"
		},
		{
			code: "indian",
			label: "Indian"
		},
		{
			code: "south_african",
			label: "South African"
		}
	],
	session: {},
	samples: bs(),
	captured: xs(),
	training: ys(),
	auto: {},
	autoForm: Cs(),
	wakeWords: [],
	wakeWordLibrary: [],
	modelHistory: [],
	managedData: Ss(),
	selectedFiles: [],
	sampleBucket: "personal",
	samplePage: {
		personal: 0,
		negative: 0
	},
	uploadProgress: 0,
	uploadLabel: "No upload in progress",
	uploadDetail: "Choose files and upload when you are ready.",
	consoleOpen: !1,
	taterLinkOpen: !1,
	trimItem: null,
	trimBucket: "personal",
	toast: {
		message: "",
		tone: "success",
		serial: 0
	}
}), ws = 0, Ts = 0, Es = J(() => Number(Y.samples.personal_count ?? Y.session.takes_received ?? 0)), Ds = J(() => Number(Y.samples.negative_count ?? Y.captured.negative_count ?? 0)), Os = J(() => Y.languages.find((e) => e.code === Y.language) || Y.languages[0]), ks = J(() => {
	let e = Os.value?.engines?.length ? Os.value.engines : ["omnivoice"], t = Y.ttsMode === "piper" ? e.filter((e) => e === "piper") : Y.ttsMode === "hybrid" ? e : e.filter((e) => e !== "piper"), n = {
		omnivoice: "OmniVoice",
		qwen3: "Qwen3",
		moss: "MOSS",
		piper: "Piper"
	}, r = Y.ttsMode === "piper" ? "Legacy" : Ps(Os.value?.quality || "experimental");
	return `${t.map((e) => n[e] || e).join(" + ") || "Unavailable"} · ${r}`;
}), As = J(() => !!(Y.training.running || Y.training.exit_code !== null || Y.training.log_lines?.length)), js = J(() => Y.samples[Y.sampleBucket] || []), Ms = J(() => !!Y.auto.trainer_link?.linked), Ns = J(() => {
	let e = Y.auto.stt_engines;
	return Array.isArray(e) && e.length ? e : [{
		id: "faster_whisper",
		label: "Faster Whisper"
	}, {
		id: "parakeet_onnx",
		label: "Parakeet ONNX"
	}];
});
function Ps(e) {
	return String(e || "").replaceAll("_", " ").replace(/\b\w/g, (e) => e.toUpperCase());
}
function X(e) {
	return e ? Y.busy.has(e) : Y.busy.size > 0;
}
function Z(e, t) {
	t ? Y.busy.add(e) : Y.busy.delete(e);
}
function Q(e, t = "success") {
	Y.toast = {
		message: String(e || ""),
		tone: t,
		serial: Y.toast.serial + 1
	};
}
function $(e, t) {
	Q(e instanceof Error ? e.message : t, "error");
}
function Fs(e) {
	Y.session = e || {}, Array.isArray(e.available_languages) && e.available_languages.length && (Y.languages = e.available_languages), Array.isArray(e.available_english_accents) && e.available_english_accents.length && (Y.englishAccents = e.available_english_accents), e.raw_phrase && (Y.phrase = e.raw_phrase), e.language && (Y.language = e.language), e.english_accent && (Y.englishAccent = e.english_accent), e.tts_mode && (Y.ttsMode = e.tts_mode), e.training && (Y.training = e.training);
}
async function Is() {
	let e = await gs("/api/session");
	return Fs(e), e;
}
async function Ls() {
	if (!Y.phrase.trim()) {
		Q("Enter a wake phrase first.", "warning");
		return;
	}
	Z("session", !0);
	try {
		let e = await _s("/api/start_session", {
			phrase: Y.phrase.trim(),
			language: Y.language,
			english_accent: Y.englishAccent,
			tts_mode: Y.ttsMode
		});
		Fs(e), await rc(!0), Q(`Session ${e.safe_word || "started"} is ready.`);
	} catch (e) {
		$(e, "Session failed to start.");
	} finally {
		Z("session", !1);
	}
}
async function Rs() {
	let e = !!Y.training.running;
	if (!(e && !window.confirm("Training is running. Stop training cleanly and end this session?"))) {
		Z("session", !0), Ts &&= (window.clearInterval(Ts), 0);
		try {
			Fs(await _s("/api/stop_session")), Q(e ? "Training stopped cleanly and the session ended." : "Session ended. You can edit the wake phrase now.");
		} catch (t) {
			e && dc(), $(t, "Session could not be stopped.");
		} finally {
			Z("session", !1);
		}
	}
}
function zs() {
	if (!Y.phrase.trim() || !("speechSynthesis" in window)) return;
	let e = new SpeechSynthesisUtterance(Y.phrase.trim());
	e.lang = Y.language, window.speechSynthesis.cancel(), window.speechSynthesis.speak(e);
}
function Bs() {
	let e = Os.value?.engines || [], t = e.some((e) => e !== "piper"), n = e.includes("piper");
	Y.ttsMode === "modern" && !t && (Y.ttsMode = "piper"), Y.ttsMode === "hybrid" && !(t && n) && (Y.ttsMode = t ? "modern" : "piper"), Y.ttsMode === "piper" && !n && (Y.ttsMode = "modern"), (Y.language !== "en" || Y.ttsMode === "piper") && (Y.englishAccent = "mixed");
}
async function Vs(e = !1) {
	e || Z("samples", !0);
	try {
		let e = await gs("/api/samples");
		Y.samples = {
			...bs(),
			...e
		};
		for (let e of ["personal", "negative"]) {
			let t = Math.max(0, Math.ceil((Y.samples[e]?.length || 0) / 50) - 1);
			Y.samplePage[e] = Math.min(Y.samplePage[e], t);
		}
		return e;
	} finally {
		e || Z("samples", !1);
	}
}
async function Hs(e = !1) {
	e || Z("captured", !0);
	try {
		let e = await gs("/api/captured_audio");
		return Y.captured = {
			...xs(),
			...e
		}, e;
	} finally {
		e || Z("captured", !1);
	}
}
function Us(e) {
	let t = e.target;
	Y.selectedFiles = Array.from(t.files || []);
}
function Ws(e, t, n) {
	return new Promise((r, i) => {
		let a = new XMLHttpRequest(), o = new FormData();
		o.append("file", e, e.name), a.open("POST", "/api/upload_personal_sample"), a.responseType = "json", a.upload.onprogress = (r) => {
			r.lengthComputable && (Y.uploadProgress = Math.round((t + r.loaded / r.total) / n * 100), Y.uploadLabel = `Uploading ${e.name} (${t + 1}/${n})`, Y.uploadDetail = "Sending and normalizing the recording.");
		}, a.onload = () => {
			let t = a.response || {};
			a.status >= 200 && a.status < 300 ? r(t) : i(Error(t.error || `Upload failed for ${e.name}`));
		}, a.onerror = () => i(/* @__PURE__ */ Error(`Upload failed for ${e.name}`)), a.send(o);
	});
}
async function Gs(e) {
	if (!Y.session.safe_word) {
		Q("Start a trainer session before uploading samples.", "warning");
		return;
	}
	if (Y.selectedFiles.length) {
		Z("upload", !0);
		try {
			let t = [...Y.selectedFiles];
			for (let e = 0; e < t.length; e += 1) await Ws(t[e], e, t.length);
			Y.uploadProgress = 100, Y.uploadLabel = "Upload complete", Y.uploadDetail = `${t.length} sample${t.length === 1 ? "" : "s"} saved in the required training format.`, Y.selectedFiles = [], e && (e.value = ""), await Promise.all([Is(), Vs(!0)]), Q("Personal samples uploaded.");
		} catch (e) {
			Y.uploadProgress = 0, $(e, "Sample upload failed.");
		} finally {
			Z("upload", !1);
		}
	}
}
async function Ks(e, t) {
	if (t === "discard" && !window.confirm(`Discard ${e.saved_as} from the captured-audio inbox?`)) return;
	let n = String(e.capture_scope || "").trim(), r = String(e.review_wake_word || e.dataset_wake_word || "").trim();
	if (t !== "discard" && !r) {
		Q("Choose the saved wake word this clip should train first.", "warning");
		return;
	}
	let i = new URLSearchParams();
	n && i.set("capture_word", n), t !== "discard" && i.set("target_word", r), Z("review", !0);
	try {
		await _s(`/api/captured_audio/${encodeURIComponent(e.saved_as)}/${t}?${i.toString()}`), await Promise.all([
			Is(),
			Hs(!0),
			Vs(!0)
		]), Q(t === "approve_personal" ? `Clip added to ${r} positives.` : t === "mark_negative" ? `Clip added to ${r} negatives.` : "Clip discarded.");
	} catch (e) {
		$(e, "Review action failed.");
	} finally {
		Z("review", !1);
	}
}
async function qs(e, t) {
	if (window.confirm(`Remove ${e.saved_as} from ${t} samples?`)) {
		Z("review", !0);
		try {
			await hs(`/api/samples/${t}/${encodeURIComponent(e.saved_as)}`, { method: "DELETE" }), await Vs(!0), Q("Sample removed.");
		} catch (e) {
			$(e, "Sample removal failed.");
		} finally {
			Z("review", !1);
		}
	}
}
async function Js(e, t) {
	if (!window.confirm(`Revert ${e.saved_as} to its pre-trim version?`)) return;
	let n = new FormData();
	n.append("bucket", t), n.append("file_name", e.saved_as), Z("review", !0);
	try {
		await hs("/api/samples/revert", {
			method: "POST",
			body: n
		}), await Vs(!0), Q("Original sample restored.");
	} catch (e) {
		$(e, "Sample revert failed.");
	} finally {
		Z("review", !1);
	}
}
async function Ys(e) {
	let t = e === "personal" ? Es.value : Ds.value;
	if (!(!t || !window.confirm(`Clear ${t} ${e} sample${t === 1 ? "" : "s"}?`))) {
		Z("review", !0);
		try {
			await _s(e === "personal" ? "/api/reset_recordings" : "/api/reset_negative_samples"), await Promise.all([
				Is(),
				Vs(!0),
				Hs(!0)
			]), Q(`${Ps(e)} samples cleared.`);
		} catch (e) {
			$(e, "Samples could not be cleared.");
		} finally {
			Z("review", !1);
		}
	}
}
function Xs(e, t) {
	Y.auto = e || {}, t && (Y.autoForm = {
		...Cs(),
		...e.config || {}
	}, Array.isArray(e.config?.generated_negative_phrases) && (Y.autoForm.generated_negative_phrases = e.config.generated_negative_phrases.join("\n")), Y.autoForm.wake_phrase || (Y.autoForm.wake_phrase = Y.session.raw_phrase || ""), Y.autoForm.language || (Y.autoForm.language = Y.session.language || "en"), Y.autoForm.english_accent || (Y.autoForm.english_accent = Y.session.english_accent || "mixed"), String(Y.autoForm.language).toLowerCase().startsWith("en") || (Y.autoForm.english_accent = "mixed"));
}
async function Zs(e = !1) {
	let t = await gs("/api/auto_train");
	return Xs(t, e), t;
}
async function Qs() {
	if (![
		Y.autoForm.train_microwakeword,
		Y.autoForm.train_openwakeword,
		Y.autoForm.train_nanowakeword_cnn,
		Y.autoForm.train_nanowakeword_gru
	].some(Boolean)) {
		Q("Select at least one model for Auto Training.", "warning");
		return;
	}
	Z("auto", !0);
	try {
		let e = await vs("/api/auto_train", Y.autoForm);
		Xs(e, !0), Q(e.config?.enabled ? "Auto Training saved and enabled." : "Auto Training saved.");
	} catch (e) {
		$(e, "Auto Training settings failed to save.");
	} finally {
		Z("auto", !1);
	}
}
async function $s(e) {
	Z("auto", !0);
	try {
		let t = await _s("/api/auto_train/action", { action: e });
		Xs(t, !1), e === "train_now" && (Y.consoleOpen = !0, dc()), Q(e === "review_now" ? `${Number(t.queued || 0)} clips queued for review.` : e === "train_now" ? "Training started." : "Wake word published.");
	} catch (e) {
		$(e, "Auto Training action failed.");
	} finally {
		Z("auto", !1);
	}
}
async function ec(e, t) {
	Z("link", !0);
	try {
		return await _s("/api/tater_link/claim", {
			tater_url: e.trim(),
			pairing_code: t.trim()
		}), Y.autoForm.tater_url = e.trim(), await Zs(!1), Q("Trainer linked securely to Tater."), !0;
	} catch (e) {
		return $(e, "Tater link failed."), !1;
	} finally {
		Z("link", !1);
	}
}
async function tc() {
	if (window.confirm("Unlink this trainer from Tater?")) {
		Z("auto", !0);
		try {
			await _s("/api/tater_link/unlink"), await Zs(!1), Q("Trainer unlinked from Tater.", "warning");
		} catch (e) {
			$(e, "Tater unlink failed.");
		} finally {
			Z("auto", !1);
		}
	}
}
async function nc(e = !1) {
	e || Z("firmware", !0);
	try {
		let e = await gs("/api/trained_wake_words/catalog");
		Y.wakeWords = Array.isArray(e.wake_words) ? e.wake_words : [];
	} finally {
		e || Z("firmware", !1);
	}
}
async function rc(e = !1) {
	e || Z("library", !0);
	try {
		let e = await gs("/api/wake_word_library");
		Y.wakeWordLibrary = Array.isArray(e.words) ? e.words : [], Y.modelHistory = (await Promise.all(Y.wakeWordLibrary.map(async (e) => {
			try {
				let t = await gs(`/api/wake_word_library/${encodeURIComponent(e.safe_word)}/history`);
				return Array.isArray(t.runs) ? t.runs : [];
			} catch {
				return [];
			}
		}))).flat().sort((e, t) => String(t.completed_at || "").localeCompare(String(e.completed_at || "")));
	} finally {
		e || Z("library", !1);
	}
}
async function ic(e) {
	if (Y.session.safe_word && Y.session.safe_word !== e.safe_word) {
		Q("Stop the current session before switching saved wake words.", "warning");
		return;
	}
	if (Y.session.safe_word === e.safe_word) {
		Q(`${e.phrase} is already active.`);
		return;
	}
	Y.phrase = e.phrase, e.language && (Y.language = e.language), e.english_accent && (Y.englishAccent = e.english_accent), e.tts_mode && (Y.ttsMode = e.tts_mode), Bs(), await Ls(), await Promise.all([Vs(!0), rc(!0)]);
}
async function ac() {
	Z("data", !0);
	try {
		let e = await gs("/api/data");
		return Y.managedData = {
			...Ss(),
			...e
		}, e;
	} finally {
		Z("data", !1);
	}
}
async function oc(e) {
	if (!e.file_count) return;
	let t = `${hc(e.size_bytes)} · ${Number(e.file_count).toLocaleString()} file${e.file_count === 1 ? "" : "s"}`, n = e.rebuild_note ? `\n\n${e.rebuild_note}` : "";
	if (window.confirm(`Permanently delete ${e.label} (${t})?${n}\n\nThis cannot be undone.`)) {
		Z("data-delete", !0);
		try {
			let t = await hs(`/api/data/${encodeURIComponent(e.id)}`, { method: "DELETE" });
			Y.managedData = {
				...Ss(),
				...t
			}, await Promise.allSettled([
				Is(),
				Vs(!0),
				Hs(!0),
				nc(!0)
			]), Q(`${e.label} deleted. ${hc(e.size_bytes)} released.`);
		} catch (t) {
			$(t, `${e.label} could not be deleted.`);
		} finally {
			Z("data-delete", !1);
		}
	}
}
async function sc(e) {
	try {
		await navigator.clipboard.writeText(e), Q("Wake-word JSON URL copied.");
	} catch (e) {
		$(e, "Clipboard unavailable.");
	}
}
async function cc() {
	if (![
		Y.trainMicroWakeWord,
		Y.trainOpenWakeWord,
		Y.trainNanoWakeWordCnn,
		Y.trainNanoWakeWordGru
	].some(Boolean)) {
		Q("Select at least one wake-word model to train.", "warning");
		return;
	}
	await Promise.all([Is(), Vs(!0)]);
	let e = !1;
	if (!(!Es.value && (e = window.confirm("No positive samples are saved. Train anyway without personal voices?"), !e))) {
		Z("training-start", !0), Y.training = {
			running: !0,
			exit_code: null,
			log_lines: ["Waiting for training output…"]
		}, Y.consoleOpen = !0;
		try {
			await _s("/api/train", {
				allow_no_personal: e,
				train_microwakeword: Y.trainMicroWakeWord,
				train_openwakeword: Y.trainOpenWakeWord,
				train_nanowakeword_cnn: Y.trainNanoWakeWordCnn,
				train_nanowakeword_gru: Y.trainNanoWakeWordGru,
				generated_negative_phrases: Y.generatedNegativePhrases
			}), dc();
		} catch (e) {
			Y.training = {
				running: !1,
				exit_code: 1,
				log_lines: [e instanceof Error ? e.message : String(e)]
			}, $(e, "Training could not start.");
		} finally {
			Z("training-start", !1);
		}
	}
}
async function lc() {
	if (!Y.session.safe_word) {
		Q("Select or start a saved wake word first.", "warning");
		return;
	}
	if (!Y.wakeWordLibrary.find((e) => e.safe_word === Y.session.safe_word)?.boundary_only_ready) {
		Q("Run one full MicroWakeWord training for this word before training its boundary only.", "warning");
		return;
	}
	Z("boundary-start", !0), Y.training = {
		running: !0,
		exit_code: null,
		log_lines: ["Preparing cached data for boundary-only training…"]
	}, Y.consoleOpen = !0;
	try {
		await _s("/api/train", {
			allow_no_personal: !0,
			train_boundary_only: !0,
			train_microwakeword: !1,
			train_openwakeword: !1,
			train_nanowakeword_cnn: !1,
			train_nanowakeword_gru: !1,
			generated_negative_phrases: Y.generatedNegativePhrases
		}), dc();
	} catch (e) {
		Y.training = {
			running: !1,
			exit_code: 1,
			log_lines: [e instanceof Error ? e.message : String(e)]
		}, $(e, "Boundary training could not start.");
	} finally {
		Z("boundary-start", !1);
	}
}
async function uc(e, t = "") {
	let n = e.toLowerCase();
	Z("nww-revalidate", !0), Y.training = {
		running: !0,
		exit_code: null,
		log_lines: [`Revalidating existing NanoWakeWord ${n.toUpperCase()} candidate…`]
	}, Y.consoleOpen = !0;
	try {
		await _s("/api/revalidate_nanowakeword", {
			architecture: n,
			safe_word: t
		}), dc();
	} catch (e) {
		Y.training = {
			running: !1,
			exit_code: 1,
			log_lines: [e instanceof Error ? e.message : String(e)]
		}, $(e, "NanoWakeWord candidate revalidation could not start.");
	} finally {
		Z("nww-revalidate", !1);
	}
}
function dc() {
	if (Ts) return;
	let e = async () => {
		try {
			let e = await gs("/api/train_status");
			Y.training = {
				...ys(),
				...e.training || {}
			}, Y.training.running || (window.clearInterval(Ts), Ts = 0, await Promise.all([
				Vs(!0),
				nc(!0),
				rc(!0)
			]), Q(Y.training.exit_code === 0 ? "Training finished successfully." : `Training ended with exit ${Y.training.exit_code}.`, Y.training.exit_code === 0 ? "success" : "error"));
		} catch {}
	};
	e(), Ts = window.setInterval(() => void e(), 1500);
}
async function fc() {
	Z("bootstrap", !0);
	try {
		await Promise.allSettled([
			Is(),
			Vs(!0),
			Hs(!0),
			Zs(!0),
			nc(!0),
			rc(!0)
		]), Bs();
		try {
			let e = await gs("/api/train_status");
			Y.training = {
				...ys(),
				...e.training || {}
			}, Y.training.running && (Y.consoleOpen = !0, dc());
		} catch {}
		ws = window.setInterval(() => {
			Y.activeView === "auto" && !X("auto") && Zs(!1).catch(() => void 0);
		}, 2500), Y.initialized = !0;
	} finally {
		Z("bootstrap", !1);
	}
}
function pc() {
	window.clearInterval(ws), window.clearInterval(Ts), ws = 0, Ts = 0;
}
function mc(e) {
	if (!e) return "";
	let t = new Date(String(e));
	return Number.isNaN(t.getTime()) ? String(e) : t.toLocaleString();
}
function hc(e) {
	let t = Math.max(0, Number(e) || 0);
	if (t < 1024) return `${Math.round(t)} B`;
	let n = [
		"KB",
		"MB",
		"GB",
		"TB"
	], r = t / 1024, i = n[0];
	for (let e = 1; e < n.length && r >= 1024; e += 1) r /= 1024, i = n[e];
	return `${r >= 10 ? r.toFixed(1) : r.toFixed(2)} ${i}`;
}
function gc(e) {
	if (!e) return "16 kHz · mono · 16-bit WAV";
	let t = Number(e.sample_rate || e.sample_rate_hz || 16e3), n = Number(e.channels || 1) === 1 ? "mono" : `${e.channels} channels`, r = Number(e.bits_per_sample || e.sample_width_bits || 16);
	return `${Math.round(t / 1e3)} kHz · ${n} · ${r}-bit`;
}
function _c(e) {
	if (e.blocked_by_vad) return {
		label: "Blocked by VAD",
		tone: "warning"
	};
	let t = String(e.event_type || "").toLowerCase();
	return t.includes("close") ? {
		label: e.capture_label || "Close miss",
		tone: "warning"
	} : t.includes("false") ? {
		label: e.capture_label || "False trigger",
		tone: "error"
	} : t.includes("wake") || t.includes("detect") ? {
		label: e.capture_label || "Wake trigger",
		tone: "success"
	} : {
		label: e.capture_label || "Captured",
		tone: "neutral"
	};
}
function vc(e, t) {
	return e.audio_url || `/api/audio/${t}/${encodeURIComponent(e.saved_as)}`;
}
//#endregion
//#region src/components/AudioTrimModal.vue?vue&type=script&setup=true&lang.ts
var yc = {
	class: "modal trim-modal",
	role: "dialog",
	"aria-modal": "true",
	"aria-label": "Trim audio"
}, bc = { class: "modal-head" }, xc = {
	key: 0,
	class: "empty-state"
}, Sc = { class: "range-grid" }, Cc = ["max"], wc = ["min", "max"], Tc = { class: "row space" }, Ec = { class: "pill" }, Dc = {
	key: 0,
	class: "pill success"
}, Oc = { class: "row modal-actions" }, kc = ["disabled"], Ac = /* @__PURE__ */ pr({
	__name: "AudioTrimModal",
	setup(e) {
		let t = /* @__PURE__ */ Gt(null), n = /* @__PURE__ */ Gt(null), r = /* @__PURE__ */ Gt(0), i = /* @__PURE__ */ Gt(0), a = /* @__PURE__ */ Gt(0), o = /* @__PURE__ */ Gt([]), s = /* @__PURE__ */ Gt(!1), c = /* @__PURE__ */ Gt(!1);
		Pn(() => Y.trimItem, async (e) => {
			if (!e) {
				n.value = null;
				return;
			}
			s.value = !0;
			try {
				let t = `/api/audio/${encodeURIComponent(Y.trimBucket)}/${encodeURIComponent(e.saved_as)}`, s = await fetch(t);
				if (!s.ok) throw Error("Audio could not be loaded.");
				let c = new (window.AudioContext || window.webkitAudioContext)();
				n.value = await c.decodeAudioData(await s.arrayBuffer()), r.value = n.value.duration, i.value = 0, a.value = r.value, await c.close();
				try {
					let t = await hs(`/api/samples/${encodeURIComponent(Y.trimBucket)}/${encodeURIComponent(e.saved_as)}/vad`, { method: "POST" });
					o.value = Array.isArray(t.segments) ? t.segments : [], o.value.length && (i.value = Math.max(0, Number(o.value[0].start || 0)), a.value = Math.min(r.value, Number(o.value[0].end || r.value)));
				} catch {
					o.value = [];
				}
				await gn(), d();
			} catch (e) {
				Q(e instanceof Error ? e.message : "Audio could not be loaded.", "error"), l();
			} finally {
				s.value = !1;
			}
		}, { immediate: !0 }), Pn([i, a], () => d());
		function l() {
			Y.trimItem = null, n.value = null, o.value = [];
		}
		function u() {
			let e = o.value[0];
			e && (i.value = Number(e.start), a.value = Number(e.end));
		}
		function d() {
			let e = t.value, s = n.value;
			if (!e || !s || !r.value) return;
			let c = e.getBoundingClientRect();
			if (!c.width || !c.height) return;
			let l = window.devicePixelRatio || 1;
			e.width = Math.round(c.width * l), e.height = Math.round(c.height * l);
			let u = e.getContext("2d");
			if (!u) return;
			u.scale(l, l);
			let d = c.width, f = c.height, p = f / 2, m = s.getChannelData(0), h = Math.max(1, Math.floor(m.length / d));
			u.clearRect(0, 0, d, f), u.strokeStyle = "rgba(222, 218, 212, .24)", u.lineWidth = 1, u.beginPath();
			for (let e = 0; e < d; e += 1) {
				let t = 1, n = -1;
				for (let r = 0; r < h; r += 1) {
					let i = m[Math.floor(e) * h + r] || 0;
					t = Math.min(t, i), n = Math.max(n, i);
				}
				u.moveTo(e, p + t * p * .84), u.lineTo(e, p + n * p * .84);
			}
			u.stroke();
			let g = i.value / r.value * d, _ = a.value / r.value * d;
			u.fillStyle = "rgba(8, 8, 9, .66)", u.fillRect(0, 0, g, f), u.fillRect(_, 0, d - _, f), u.fillStyle = "rgba(255, 145, 52, .12)", u.fillRect(g, 0, _ - g, f), u.strokeStyle = "#ff9134", u.lineWidth = 2;
			for (let e of [g, _]) u.beginPath(), u.moveTo(e, 0), u.lineTo(e, f), u.stroke();
			u.strokeStyle = "rgba(68, 225, 165, .55)";
			for (let e of o.value) {
				let t = e.start / r.value * d;
				u.beginPath(), u.moveTo(t, 0), u.lineTo(t, f), u.stroke();
			}
		}
		function f() {
			let e = n.value;
			if (!e) return;
			let t = new (window.AudioContext || window.webkitAudioContext)(), r = t.createBufferSource();
			r.buffer = e, r.connect(t.destination), r.start(0, i.value, Math.max(.01, a.value - i.value)), r.onended = () => void t.close();
		}
		async function p() {
			let e = n.value;
			if (!e) throw Error("Audio is not loaded.");
			let t = Math.floor(i.value * e.sampleRate), r = Math.min(Math.floor(a.value * e.sampleRate), e.length), o = 16e3, s;
			if (e.sampleRate === o) s = e.getChannelData(0).slice(t, r);
			else {
				let n = Math.max(1, Math.floor((r - t) * o / e.sampleRate)), c = new OfflineAudioContext(1, n, o), l = c.createBufferSource();
				l.buffer = e, l.connect(c.destination), l.start(0, i.value, a.value - i.value), s = (await c.startRendering()).getChannelData(0);
			}
			let c = /* @__PURE__ */ new ArrayBuffer(44 + s.length * 2), l = new DataView(c);
			l.setUint32(0, 1380533830, !1), l.setUint32(4, 36 + s.length * 2, !0), l.setUint32(8, 1463899717, !1), l.setUint32(12, 1718449184, !1), l.setUint32(16, 16, !0), l.setUint16(20, 1, !0), l.setUint16(22, 1, !0), l.setUint32(24, o, !0), l.setUint32(28, o * 2, !0), l.setUint16(32, 2, !0), l.setUint16(34, 16, !0), l.setUint32(36, 1684108385, !1), l.setUint32(40, s.length * 2, !0);
			for (let e = 0; e < s.length; e += 1) l.setInt16(44 + e * 2, Math.max(-32768, Math.min(32767, Math.round(s[e] * 32767))), !0);
			return new Blob([c], { type: "audio/wav" });
		}
		async function m() {
			let e = Y.trimItem;
			if (e) {
				c.value = !0;
				try {
					let t = new FormData();
					t.append("file", await p(), "trimmed.wav"), t.append("bucket", Y.trimBucket), t.append("source_file", e.saved_as), t.append("start_time", i.value.toFixed(3)), t.append("end_time", a.value.toFixed(3));
					let n = await hs("/api/samples/trim", {
						method: "POST",
						body: t
					});
					l(), await Vs(!0), Q(n.message || "Trimmed sample saved.");
				} catch (e) {
					Q(e instanceof Error ? e.message : "Trim failed.", "error");
				} finally {
					c.value = !1;
				}
			}
		}
		function h() {
			Y.trimItem && d();
		}
		return window.addEventListener("resize", h), jr(() => window.removeEventListener("resize", h)), (e, n) => (H(), ia(Yn, { to: "body" }, [I(Y).trimItem ? (H(), U("div", {
			key: 0,
			class: "modal-backdrop",
			onClick: os(l, ["self"])
		}, [W("section", yc, [
			W("header", bc, [W("div", null, [n[2] ||= W("span", { class: "eyebrow" }, "Audio editor", -1), W("h2", null, "Trim " + A(I(Y).trimItem.saved_as), 1)]), W("button", {
				type: "button",
				class: "button ghost",
				onClick: l
			}, "Close")]),
			n[3] ||= W("p", { class: "muted" }, "Keep the spoken wake phrase and remove excess silence or noise. VAD markers appear in green.", -1),
			s.value ? (H(), U("div", xc, "Loading waveform…")) : (H(), U(V, { key: 1 }, [
				W("canvas", {
					ref_key: "canvas",
					ref: t,
					class: "waveform"
				}, null, 512),
				W("div", Sc, [W("label", null, [W("span", null, "Start · " + A(i.value.toFixed(2)) + "s", 1), L(W("input", {
					"onUpdate:modelValue": n[0] ||= (e) => i.value = e,
					type: "range",
					min: "0",
					max: Math.max(0, a.value - .01),
					step: ".01"
				}, null, 8, Cc), [[
					Zo,
					i.value,
					void 0,
					{ number: !0 }
				]])]), W("label", null, [W("span", null, "End · " + A(a.value.toFixed(2)) + "s", 1), L(W("input", {
					"onUpdate:modelValue": n[1] ||= (e) => a.value = e,
					type: "range",
					min: Math.min(r.value, i.value + .01),
					max: r.value,
					step: ".01"
				}, null, 8, wc), [[
					Zo,
					a.value,
					void 0,
					{ number: !0 }
				]])])]),
				W("div", Tc, [W("span", Ec, "Selection " + A(Math.max(0, a.value - i.value).toFixed(2)) + "s", 1), o.value.length ? (H(), U("span", Dc, A(o.value.length) + " speech segment" + A(o.value.length === 1 ? "" : "s"), 1)) : q("", !0)]),
				W("div", Oc, [
					W("button", {
						type: "button",
						onClick: f
					}, "Play selection"),
					o.value.length ? (H(), U("button", {
						key: 0,
						type: "button",
						onClick: u
					}, "Select first VAD")) : q("", !0),
					W("button", {
						type: "button",
						class: "button primary",
						disabled: c.value,
						onClick: m
					}, A(c.value ? "Saving…" : "Save trim"), 9, kc)
				])
			], 64))
		])])) : q("", !0)]));
	}
}), jc = { class: "app-shell" }, Mc = { class: "app-header" }, Nc = { class: "header-status" }, Pc = {
	key: 0,
	class: "session-chip"
}, Fc = {
	class: "tabs",
	"aria-label": "Trainer areas"
}, Ic = ["onClick"], Lc = { class: "tab-full" }, Rc = { class: "tab-short" }, zc = { key: 0 }, Bc = { class: "main-content" }, Vc = {
	key: 0,
	class: "loading-panel"
}, Hc = { class: "panel word-library-panel" }, Uc = { class: "panel-head" }, Wc = {
	key: 0,
	class: "word-library-grid"
}, Gc = { class: "library-monogram" }, Kc = { key: 0 }, qc = { class: "library-stats" }, Jc = ["disabled", "onClick"], Yc = {
	key: 1,
	class: "library-empty"
}, Xc = {
	key: 2,
	class: "library-hint"
}, Zc = { class: "panel" }, Qc = { class: "panel-head" }, $c = { class: "form-grid phrase-form" }, el = { class: "field wide" }, tl = ["disabled"], nl = { class: "field" }, rl = ["disabled"], il = ["value"], al = {
	key: 0,
	class: "field"
}, ol = ["disabled"], sl = ["value"], cl = { class: "field" }, ll = ["disabled"], ul = ["disabled"], dl = ["disabled"], fl = ["disabled"], pl = { class: "row form-actions" }, ml = ["disabled"], hl = ["disabled"], gl = ["disabled"], _l = { class: "panel" }, vl = { class: "panel-head" }, yl = { class: "stats" }, bl = { class: "model-picker" }, xl = {
	class: "model-choice-grid",
	role: "group",
	"aria-label": "Wake-word models to train"
}, Sl = ["aria-pressed"], Cl = ["aria-pressed"], wl = ["aria-pressed"], Tl = {
	key: 0,
	class: "nano-architecture-panel"
}, El = { class: "nano-architecture-heading" }, Dl = {
	class: "nano-architecture-grid",
	role: "group",
	"aria-label": "NanoWakeWord architectures"
}, Ol = ["aria-pressed"], kl = ["aria-pressed"], Al = {
	key: 1,
	class: "hard-negative-card"
}, jl = { class: "hard-negative-head" }, Ml = { class: "hard-negative-engines" }, Nl = { class: "field" }, Pl = {
	key: 0,
	class: "training-summary-card"
}, Fl = { class: "training-summary-grid" }, Il = {
	key: 0,
	class: "summary-metrics"
}, Ll = { class: "train-action" }, Rl = ["disabled"], zl = ["disabled"], Bl = ["disabled"], Vl = { class: "panel-footer" }, Hl = ["disabled"], Ul = { class: "hero auto-hero" }, Wl = { class: "panel" }, Gl = { class: "toggle-list" }, Kl = { class: "form-grid" }, ql = { class: "field" }, Jl = { class: "field" }, Yl = {
	key: 0,
	class: "field"
}, Xl = ["value"], Zl = { class: "field wide" }, Ql = ["value"], $l = { class: "field" }, eu = { class: "panel" }, tu = { class: "form-grid" }, nu = { class: "field" }, ru = { class: "field" }, iu = { class: "model-picker" }, au = {
	class: "model-choice-grid",
	role: "group",
	"aria-label": "Auto Training wake-word models"
}, ou = ["aria-pressed"], su = ["aria-pressed"], cu = ["aria-pressed"], lu = {
	key: 0,
	class: "nano-architecture-panel"
}, uu = { class: "nano-architecture-heading" }, du = {
	class: "nano-architecture-grid",
	role: "group",
	"aria-label": "Auto Training NanoWakeWord architectures"
}, fu = ["aria-pressed"], pu = ["aria-pressed"], mu = {
	key: 1,
	class: "hard-negative-card"
}, hu = { class: "hard-negative-head" }, gu = { class: "hard-negative-engines" }, _u = { class: "field" }, vu = {
	key: 0,
	class: "training-summary-card"
}, yu = { class: "training-summary-grid" }, bu = {
	key: 0,
	class: "summary-metrics"
}, xu = { class: "stats" }, Su = { class: "format-value" }, Cu = { class: "format-value" }, wu = { class: "panel" }, Tu = { class: "form-grid" }, Eu = { class: "field wide" }, Du = { class: "field wide" }, Ou = { class: "link-row" }, ku = ["disabled"], Au = ["disabled"], ju = { class: "toggle-list compact" }, Mu = { class: "panel action-panel" }, Nu = { class: "action-grid" }, Pu = ["disabled"], Fu = ["disabled"], Iu = ["disabled"], Lu = ["disabled"], Ru = { class: "audit" }, zu = { class: "hero capture-hero" }, Bu = { class: "panel" }, Vu = { class: "panel-head" }, Hu = ["disabled"], Uu = { class: "stats" }, Wu = { class: "panel" }, Gu = {
	key: 0,
	class: "empty-state"
}, Ku = {
	key: 1,
	class: "audio-list"
}, qu = {
	key: 0,
	class: "meta-row"
}, Ju = ["onUpdate:modelValue"], Yu = ["value"], Xu = {
	key: 1,
	class: "transcript"
}, Zu = {
	key: 2,
	class: "transcript"
}, Qu = ["src"], $u = ["disabled", "onClick"], ed = ["disabled", "onClick"], td = ["disabled", "onClick"], nd = { class: "hero samples-hero" }, rd = { class: "pill hero-pill" }, id = { class: "panel" }, ad = { class: "panel-head sample-head" }, od = { class: "segment-control" }, sd = { class: "row toolbar" }, cd = ["disabled"], ld = ["disabled"], ud = ["disabled"], dd = {
	key: 0,
	class: "empty-state"
}, fd = {
	key: 1,
	class: "audio-list compact-list"
}, pd = { class: "row" }, md = {
	key: 0,
	class: "pill warning"
}, hd = {
	key: 0,
	class: "transcript"
}, gd = {
	key: 1,
	class: "transcript"
}, _d = ["src"], vd = ["onClick"], yd = ["onClick"], bd = ["disabled", "onClick"], xd = {
	key: 2,
	class: "pagination"
}, Sd = ["disabled"], Cd = ["disabled"], wd = { class: "panel" }, Td = { class: "dropzone" }, Ed = ["disabled"], Dd = { class: "progress-card" }, Od = { class: "progress-track" }, kd = { class: "hero data-hero" }, Ad = { class: "pill hero-pill" }, jd = { class: "panel" }, Md = { class: "panel-head" }, Nd = ["disabled"], Pd = { class: "stats" }, Fd = { class: "format-value" }, Id = {
	key: 0,
	class: "data-warning"
}, Ld = { class: "panel-head" }, Rd = { class: "number" }, zd = { class: "data-list" }, Bd = { class: "data-copy" }, Vd = { class: "data-title" }, Hd = {
	key: 0,
	class: "data-note"
}, Ud = { class: "data-usage" }, Wd = ["disabled", "onClick"], Gd = {
	key: 0,
	class: "panel empty-state"
}, Kd = { class: "hero firmware-hero" }, qd = { class: "panel" }, Jd = { class: "panel-head" }, Yd = ["disabled"], Xd = {
	key: 0,
	class: "empty-state"
}, Zd = {
	key: 1,
	class: "word-list"
}, Qd = ["href"], $d = {
	key: 1,
	class: "muted"
}, ef = ["href"], tf = ["href"], nf = ["href"], rf = ["href"], af = { class: "meta-row" }, of = { key: 0 }, sf = { key: 1 }, cf = { key: 2 }, lf = { key: 3 }, uf = ["disabled", "onClick"], df = { class: "panel history-panel" }, ff = { class: "panel-head" }, pf = {
	key: 0,
	class: "history-list"
}, mf = { class: "history-rail" }, hf = { class: "history-body" }, gf = { class: "history-models" }, _f = {
	key: 1,
	class: "empty-state"
}, vf = { class: "panel compatibility-panel" }, yf = {
	key: 0,
	class: "empty-state"
}, bf = {
	key: 1,
	class: "word-list"
}, xf = ["href"], Sf = {
	key: 1,
	class: "muted"
}, Cf = ["disabled", "onClick"], wf = {
	class: "modal console-modal",
	role: "dialog",
	"aria-modal": "true",
	"aria-label": "Training console"
}, Tf = { class: "modal-head" }, Ef = { class: "row console-actions" }, Df = { class: "console-progress-head" }, Of = { class: "console-progress-track" }, kf = {
	key: 0,
	class: "console-progress-facts"
}, Af = {
	class: "modal link-modal",
	role: "dialog",
	"aria-modal": "true",
	"aria-label": "Link Tater"
}, jf = { class: "modal-head" }, Mf = {
	key: 0,
	class: "link-success"
}, Nf = {
	key: 1,
	class: "stack"
}, Pf = { class: "field" }, Ff = { class: "field" }, If = ["disabled"], Lf = "/static/images/tater-wake-word-trainer.png", Rf = 50, zf = /* @__PURE__ */ pr({
	__name: "TrainerApp",
	setup(e) {
		let t = /* @__PURE__ */ Gt(null), n = /* @__PURE__ */ Gt(null), r = /* @__PURE__ */ Gt(!0), i = /* @__PURE__ */ Gt(""), a = /* @__PURE__ */ Gt(""), o = /* @__PURE__ */ Gt(!1), s = [
			{
				id: "trainer",
				label: "Trainer",
				short: "Train"
			},
			{
				id: "auto",
				label: "Auto Training",
				short: "Auto"
			},
			{
				id: "firmware",
				label: "Wake Words",
				short: "Words"
			},
			{
				id: "captured",
				label: "Captured Audio",
				short: "Inbox"
			},
			{
				id: "samples",
				label: "Samples",
				short: "Samples"
			},
			{
				id: "data",
				label: "Data",
				short: "Data"
			}
		], c = J(() => {
			let e = Y.samplePage[Y.sampleBucket];
			return js.value.slice(e * Rf, (e + 1) * Rf);
		}), l = J(() => Math.max(1, Math.ceil(js.value.length / Rf))), u = J(() => Y.auto.state || {}), d = J(() => Y.auto.runtime || {}), f = J(() => {
			let e = u.value, t = [];
			return e.last_review_result && t.push(`Last review: ${String(e.last_review_result).replaceAll("_", " ")}`), e.last_review_file && t.push(String(e.last_review_file)), e.last_review_transcript && t.push(`STT: “${e.last_review_transcript}”`), e.last_review_error && t.push(`Error: ${e.last_review_error}`), e.last_stt_engine && t.push(`STT engine: ${String(e.last_stt_engine).replaceAll("_", " ")}`), e.last_notify_at && t.push(e.last_notify_error ? `Publish failed: ${e.last_notify_error}` : `Wake word published ${mc(e.last_notify_at)}`), t.join(" · ") || "No automatic review has run yet.";
		}), p = J(() => Y.training.running ? {
			text: "Training running",
			tone: "warning"
		} : Y.training.exit_code === 0 ? {
			text: "Training finished",
			tone: "success"
		} : Y.training.exit_code === null ? {
			text: "Not started",
			tone: "neutral"
		} : {
			text: `Exit ${Y.training.exit_code}`,
			tone: "error"
		}), m = J(() => d.value.review_running ? {
			text: `Transcribing ${d.value.review_file || "wake"}`,
			tone: "warning"
		} : Y.training.running && Y.auto.config?.enabled ? {
			text: "Training running",
			tone: "warning"
		} : Y.auto.config?.enabled ? {
			text: "Enabled",
			tone: "success"
		} : {
			text: "Disabled",
			tone: "neutral"
		}), h = J(() => Y.training.log_lines?.length ? Y.training.log_lines : ["No training output yet."]), g = J(() => Y.training.progress || null), _ = J(() => {
			let e = String(g.value?.stage || "pipeline").split(":", 1)[0];
			return {
				starting: "Getting ready",
				datasets: "Training data",
				tts: "Speech generation",
				features: "Feature building",
				model: "Model training",
				boundary: "Wake boundary",
				calibration: "Calibration",
				package: "Packaging",
				openwakeword: "openWakeWord",
				nanowakeword: "NanoWakeWord",
				complete: "Finished",
				failed: "Needs attention",
				worker: "Pipeline progress",
				pipeline: "Live pipeline"
			}[e] || e.replaceAll("-", " ");
		}), v = J(() => {
			let e = g.value?.percent;
			if (e == null) return null;
			let t = Number(e);
			return Number.isFinite(t) ? Math.max(0, Math.min(100, t)) : null;
		}), y = J(() => {
			let e = g.value;
			if (!e) return [];
			let t = [], n = Number(e.current), r = Number(e.total), i = String(e.unit || "items");
			return Number.isFinite(n) && Number.isFinite(r) && r > 0 ? t.push(`${Math.round(n).toLocaleString()} / ${Math.round(r).toLocaleString()} ${i}`) : Number.isFinite(n) && n > 0 && t.push(`${Math.round(n).toLocaleString()} ${i}`), e.elapsed && t.push(`Elapsed ${e.elapsed}`), e.eta && t.push(`About ${e.eta} left`), e.rate_text ? t.push(String(e.rate_text)) : Number(e.rate) > 0 && t.push(`${Number(e.rate).toLocaleString()} ${i}/sec`), t;
		}), b = J(() => (Array.isArray(Y.training.summary?.models) ? Y.training.summary.models : []).find((e) => e.failure_kind === "quality_rejected" && String(e.id || "").startsWith("nano_"))), x = J(() => Y.wakeWordLibrary.find((e) => e.safe_word === Y.session.safe_word)), S = J(() => Y.modelHistory.slice(0, 16)), C = J(() => {
			let e = /* @__PURE__ */ new Map();
			for (let t of Y.managedData.items || []) {
				let n = e.get(t.category) || [];
				n.push(t), e.set(t.category, n);
			}
			return Array.from(e, ([e, t]) => ({
				name: e,
				items: t
			}));
		}), w = J(() => Y.trainNanoWakeWordCnn || Y.trainNanoWakeWordGru), T = J(() => Y.autoForm.train_nanowakeword_cnn || Y.autoForm.train_nanowakeword_gru), ee = J(() => [
			...Y.trainMicroWakeWord ? ["MWW"] : [],
			...Y.trainOpenWakeWord ? ["OWW"] : [],
			...w.value ? ["NWW"] : []
		]), te = J(() => [
			...Y.autoForm.train_microwakeword ? ["MWW"] : [],
			...Y.autoForm.train_openwakeword ? ["OWW"] : [],
			...T.value ? ["NWW"] : []
		]);
		Pn([() => Y.language, () => Y.ttsMode], Bs), Pn(() => Y.toast.serial, () => window.setTimeout(() => {
			Y.toast.message = "";
		}, 4500)), Pn(h, async () => {
			r.value && (await gn(), r.value && n.value && (n.value.scrollTop = n.value.scrollHeight));
		}), Pn(() => Y.consoleOpen, async (e) => {
			e && (r.value = !0, await gn(), D());
		}), Or(() => {
			fc(), document.addEventListener("keydown", E);
		}), jr(() => {
			pc(), document.removeEventListener("keydown", E);
		});
		function E(e) {
			e.key === "Escape" && (Y.consoleOpen = !1, Y.taterLinkOpen = !1, Y.trimItem = null);
		}
		function ne() {
			let e = n.value;
			if (!e) return;
			let t = e.scrollHeight - e.clientHeight - e.scrollTop;
			r.value = t <= 32;
		}
		function D() {
			let e = n.value;
			e && (r.value = !0, e.scrollTop = e.scrollHeight);
		}
		function re(e) {
			Y.activeView = e, (e === "auto" ? Zs(!1) : e === "captured" ? Hs() : e === "samples" ? Vs() : e === "firmware" ? Promise.all([nc(), rc()]) : e === "data" ? ac() : Promise.resolve()).catch((e) => Q(e instanceof Error ? e.message : "Refresh failed.", "error"));
		}
		function ie(e) {
			Y.sampleBucket = e;
		}
		function ae(e, t) {
			Y.trimBucket = t, Y.trimItem = e;
		}
		function oe() {
			w.value ? (Y.trainNanoWakeWordCnn = !1, Y.trainNanoWakeWordGru = !1) : Y.trainNanoWakeWordCnn = !0;
		}
		function O() {
			T.value ? (Y.autoForm.train_nanowakeword_cnn = !1, Y.autoForm.train_nanowakeword_gru = !1) : Y.autoForm.train_nanowakeword_cnn = !0;
		}
		function se(e) {
			return new Set(String(e || "").split(/[\n,]+/).map((e) => e.trim().toLocaleLowerCase()).filter(Boolean)).size;
		}
		function ce() {
			i.value = Y.autoForm.tater_url || "http://127.0.0.1:8501", a.value = "", o.value = !1, Y.taterLinkOpen = !0, gn(() => document.querySelector("#pairing-code")?.focus());
		}
		function le() {
			let e = a.value.toUpperCase().replace(/[^A-Z0-9]/g, "").slice(0, 8);
			a.value = e.length > 4 ? `${e.slice(0, 4)}-${e.slice(4)}` : e;
		}
		async function ue() {
			if (!i.value.trim() || !a.value.trim()) {
				Q("Tater address and pairing code are required.", "warning");
				return;
			}
			o.value = await ec(i.value, a.value);
		}
		function fe(e) {
			let t = [];
			return e.source_device && t.push(String(e.source_device)), e.wake_word && t.push(String(e.wake_word)), e.max_probability !== null && e.max_probability !== void 0 && t.push(`max ${e.max_probability}`), e.average_probability !== null && e.average_probability !== void 0 && t.push(`avg ${e.average_probability}`), e.detection_profile && t.push(`profile ${String(e.detection_profile).replaceAll("_", " ")}`), e.auto_review_status && t.push(`auto ${String(e.auto_review_status).replaceAll("_", " ")}`), e.vad_max_probability !== null && e.vad_max_probability !== void 0 && t.push(`VAD ${e.vad_max_probability}`), t;
		}
		function pe(e) {
			let t = [];
			e.original_name && e.original_name !== e.saved_as && t.push(`From ${e.original_name}`);
			let n = mc(e.reviewed_at || e.received_at || e.created_at);
			return n && t.push(`Saved ${n}`), e.message && t.push(String(e.message)), e.auto_negative && t.push("Auto-reviewed false positive"), e.auto_positive && t.push("Auto-promoted close miss"), t.join(" · ") || "Training sample";
		}
		function me(e) {
			return String(e.json_url || e.url || e.jsonUrl || "");
		}
		function he(e) {
			return String(e.esphome_json_url || e.esphomeJsonUrl || "");
		}
		function ge(e) {
			return String(e.model_url || e.modelUrl || "");
		}
		function _e(e) {
			return String(e.openwakeword_model_url || "");
		}
		function ve(e) {
			return String(e.wake_boundary_model_url || "");
		}
		function ye(e) {
			return e.nanowakeword_models && typeof e.nanowakeword_models == "object" ? e.nanowakeword_models : {};
		}
		function be(e) {
			let t = e.trim().toLowerCase();
			return /^(✓|✅)|success|finished/.test(t) ? "success" : /^(✗|❌)|error|failed|traceback/.test(t) ? "error" : /^(⚠|warning)/.test(t) ? "warning" : /^={4,}|^-----|^=====/.test(t) ? "heading" : "";
		}
		return (e, d) => (H(), U("div", jc, [
			d[172] ||= W("div", {
				class: "ambient ambient-one",
				"aria-hidden": "true"
			}, null, -1),
			d[173] ||= W("div", {
				class: "ambient ambient-two",
				"aria-hidden": "true"
			}, null, -1),
			W("header", Mc, [W("div", { class: "brand" }, [W("div", {
				class: "brand-mark",
				"aria-hidden": "true"
			}, [W("img", {
				src: Lf,
				alt: ""
			})]), d[58] ||= W("div", null, [
				W("span", { class: "eyebrow" }, "Tater tools"),
				W("h1", null, "Wake Word Studio"),
				W("p", null, "Generate voices, curate real recordings, train, and publish.")
			], -1)]), W("div", Nc, [W("div", { class: "local-trainer-status" }, [d[59] ||= W("span", { class: "live-dot" }, [W("i"), K("Local trainer")], -1), W("small", null, A("v47.7"))]), I(Y).session.safe_word ? (H(), U("span", Pc, A(I(Y).session.safe_word) + " · " + A(I(Y).language), 1)) : q("", !0)])]),
			W("nav", Fc, [(H(), U(V, null, R(s, (e) => W("button", {
				key: e.id,
				type: "button",
				class: k({ active: I(Y).activeView === e.id }),
				onClick: (t) => re(e.id)
			}, [
				W("span", Lc, A(e.label), 1),
				W("span", Rc, A(e.short), 1),
				e.id === "captured" && I(Y).captured.captured_count ? (H(), U("b", zc, A(I(Y).captured.captured_count), 1)) : q("", !0)
			], 10, Ic)), 64))]),
			W("main", Bc, [I(Y).initialized ? (H(), U(V, { key: 1 }, [I(Y).activeView === "trainer" ? (H(), U(V, { key: 0 }, [
				d[94] ||= fa("<section class=\"hero training-hero\"><div><span class=\"eyebrow\">Training studio</span><h2>Build a personal wake word</h2><p>Choose a multilingual voice route, check your real samples, then follow the model pipeline live.</p></div><div class=\"step-row\"><span><b>1</b> Phrase</span><span><b>2</b> Samples</span><span><b>3</b> Train</span></div></section>", 1),
				W("section", Hc, [
					W("header", Uc, [
						d[61] ||= W("div", { class: "number" }, "↺", -1),
						d[62] ||= W("div", null, [W("h3", null, "Saved wake words"), W("p", null, "Switch words without regenerating their voices, features, or reviewed samples.")], -1),
						W("span", { class: k(["pill", I(Y).wakeWordLibrary.length ? "success" : ""]) }, A(I(Y).wakeWordLibrary.length) + " saved", 3)
					]),
					I(Y).wakeWordLibrary.length ? (H(), U("div", Wc, [(H(!0), U(V, null, R(I(Y).wakeWordLibrary, (e) => (H(), U("article", {
						key: e.safe_word,
						class: k({ active: e.safe_word === I(Y).session.safe_word })
					}, [
						W("header", null, [
							W("div", Gc, A(e.phrase.slice(0, 2).toUpperCase()), 1),
							W("div", null, [W("span", null, A(e.active ? "Working set" : "Saved word"), 1), W("strong", null, A(e.phrase), 1)]),
							e.has_published_bundle ? (H(), U("i", Kc, "Ready")) : q("", !0)
						]),
						W("div", qc, [
							W("span", null, [W("b", null, A(Number(e.dataset_file_count || 0).toLocaleString()), 1), d[63] ||= K(" cached files", -1)]),
							W("span", null, [W("b", null, A(I(hc)(e.dataset_size_bytes)), 1), d[64] ||= K(" retained", -1)]),
							W("span", null, [W("b", null, A(e.run_count), 1), d[65] ||= K(" runs", -1)])
						]),
						W("footer", null, [W("small", null, [K(A(e.language || "language saved on next use"), 1), e.latest_run?.completed_at ? (H(), U(V, { key: 0 }, [K(" · latest " + A(I(mc)(e.latest_run.completed_at)), 1)], 64)) : q("", !0)]), W("button", {
							type: "button",
							class: k({ primary: e.safe_word !== I(Y).session.safe_word }),
							disabled: !!I(Y).session.safe_word && e.safe_word !== I(Y).session.safe_word,
							onClick: (t) => I(ic)(e)
						}, A(e.safe_word === I(Y).session.safe_word ? "Active" : "Use + retrain"), 11, Jc)])
					], 2))), 128))])) : (H(), U("div", Yc, [...d[66] ||= [
						W("div", null, "＋", -1),
						W("strong", null, "Your first word will be saved automatically", -1),
						W("span", null, "Its generated corpus and feature cache stay available for later retraining.", -1)
					]])),
					I(Y).session.safe_word && I(Y).wakeWordLibrary.some((e) => e.safe_word !== I(Y).session.safe_word) ? (H(), U("p", Xc, "End the current session before selecting a different saved word. Nothing is deleted when you switch.")) : q("", !0)
				]),
				W("section", Zc, [
					W("header", Qc, [
						d[67] ||= W("div", { class: "number" }, "1", -1),
						d[68] ||= W("div", null, [W("h3", null, "Phrase + voice"), W("p", null, "The phrase and voice route lock while a session is active.")], -1),
						W("span", { class: k(["pill", I(Y).session.safe_word ? "success" : ""]) }, A(I(Y).session.safe_word ? `Session · ${I(Y).session.safe_word}` : "No session"), 3)
					]),
					W("div", $c, [
						W("label", el, [d[69] ||= W("span", null, "Wake phrase", -1), L(W("input", {
							"onUpdate:modelValue": d[0] ||= (e) => I(Y).phrase = e,
							type: "text",
							placeholder: "e.g. \"hey tater\"",
							disabled: !!I(Y).session.safe_word || I(X)("session"),
							onKeyup: d[1] ||= cs((...e) => I(Ls) && I(Ls)(...e), ["enter"])
						}, null, 40, tl), [[Zo, I(Y).phrase]])]),
						W("label", nl, [
							d[70] ||= W("span", null, "Language", -1),
							L(W("select", {
								"onUpdate:modelValue": d[2] ||= (e) => I(Y).language = e,
								disabled: !!I(Y).session.safe_word || I(X)("session")
							}, [(H(!0), U(V, null, R(I(Y).languages, (e) => (H(), U("option", {
								key: e.code,
								value: e.code
							}, A(e.label), 9, il))), 128))], 8, rl), [[es, I(Y).language]]),
							W("small", null, A(I(ks)), 1)
						]),
						I(Y).language === "en" && I(Y).ttsMode !== "piper" ? (H(), U("label", al, [
							d[71] ||= W("span", null, "English accent emphasis", -1),
							L(W("select", {
								"onUpdate:modelValue": d[3] ||= (e) => I(Y).englishAccent = e,
								disabled: !!I(Y).session.safe_word || I(X)("session")
							}, [(H(!0), U(V, null, R(I(Y).englishAccents, (e) => (H(), U("option", {
								key: e.code,
								value: e.code
							}, A(e.label), 9, sl))), 128))], 8, ol), [[es, I(Y).englishAccent]]),
							d[72] ||= W("small", null, "Qwen emphasizes this accent; MOSS carries it through accepted references. OmniVoice and Piper keep broad English coverage.", -1)
						])) : q("", !0),
						W("label", cl, [
							d[73] ||= W("span", null, "TTS source", -1),
							L(W("select", {
								"onUpdate:modelValue": d[4] ||= (e) => I(Y).ttsMode = e,
								disabled: !!I(Y).session.safe_word || I(X)("session")
							}, [
								W("option", {
									value: "hybrid",
									disabled: !I(Y).languages.find((e) => e.code === I(Y).language)?.engines?.includes("piper")
								}, "Four-provider ensemble · recommended", 8, ul),
								W("option", {
									value: "modern",
									disabled: !I(Y).languages.find((e) => e.code === I(Y).language)?.engines?.some((e) => e !== "piper")
								}, "Modern only · no Piper", 8, dl),
								W("option", {
									value: "piper",
									disabled: !I(Y).languages.find((e) => e.code === I(Y).language)?.engines?.includes("piper")
								}, "Piper only · legacy", 8, fl)
							], 8, ll), [[es, I(Y).ttsMode]]),
							d[74] ||= W("small", null, "Models download once and stay cached.", -1)
						])
					]),
					W("div", pl, [I(Y).session.safe_word ? (H(), U("button", {
						key: 1,
						type: "button",
						class: "button danger",
						disabled: I(X)("session"),
						onClick: d[6] ||= (...e) => I(Rs) && I(Rs)(...e)
					}, A(I(X)("session") ? "Stopping…" : I(Y).training.running ? "Stop session + training" : "Stop session"), 9, hl)) : (H(), U("button", {
						key: 0,
						type: "button",
						class: "button primary",
						disabled: I(X)("session") || !I(Y).phrase.trim(),
						onClick: d[5] ||= (...e) => I(Ls) && I(Ls)(...e)
					}, A(I(X)("session") ? "Starting…" : "Start session"), 9, ml)), W("button", {
						type: "button",
						disabled: !I(Y).phrase.trim(),
						onClick: d[7] ||= (...e) => I(zs) && I(zs)(...e)
					}, "System preview", 8, gl)])
				]),
				W("section", _l, [
					W("header", vl, [
						d[75] ||= W("div", { class: "number" }, "2", -1),
						d[76] ||= W("div", null, [W("h3", null, "Train wake word"), W("p", null, "Personal positives and reviewed false-wake negatives are automatically included.")], -1),
						W("span", { class: k(["pill", p.value.tone]) }, A(p.value.text), 3)
					]),
					W("div", yl, [
						W("article", null, [d[77] ||= W("span", null, "Positive samples", -1), W("strong", null, A(I(Es)), 1)]),
						W("article", null, [d[78] ||= W("span", null, "Negative samples", -1), W("strong", null, A(I(Ds)), 1)]),
						d[79] ||= W("article", null, [W("span", null, "Training format"), W("strong", { class: "format-value" }, "16 kHz · mono · WAV")], -1)
					]),
					W("div", bl, [
						W("div", xl, [
							W("button", {
								type: "button",
								class: k(["model-choice", { active: I(Y).trainMicroWakeWord }]),
								"aria-pressed": I(Y).trainMicroWakeWord,
								onClick: d[8] ||= (e) => I(Y).trainMicroWakeWord = !I(Y).trainMicroWakeWord
							}, [...d[80] ||= [
								W("i", null, "MWW", -1),
								W("span", null, [W("strong", null, "MicroWakeWord"), W("small", null, "TFLite for ESP32 satellites and ESPHome.")], -1),
								W("b", null, "✓", -1)
							]], 10, Sl),
							W("button", {
								type: "button",
								class: k(["model-choice", { active: I(Y).trainOpenWakeWord }]),
								"aria-pressed": I(Y).trainOpenWakeWord,
								onClick: d[9] ||= (e) => I(Y).trainOpenWakeWord = !I(Y).trainOpenWakeWord
							}, [...d[81] ||= [
								W("i", null, "OWW", -1),
								W("span", null, [W("strong", null, "openWakeWord"), W("small", null, "ONNX for satellites with OWW support.")], -1),
								W("b", null, "✓", -1)
							]], 10, Cl),
							W("button", {
								type: "button",
								class: k(["model-choice", { active: w.value }]),
								"aria-pressed": w.value,
								onClick: oe
							}, [...d[82] ||= [
								W("i", null, "NWW", -1),
								W("span", null, [W("strong", null, "NanoWakeWord"), W("small", null, "ONNX with selectable CNN and GRU builds.")], -1),
								W("b", null, "✓", -1)
							]], 10, wl)
						]),
						w.value ? (H(), U("div", Tl, [W("div", El, [d[83] ||= W("div", null, [W("strong", null, "NanoWakeWord architectures"), W("small", null, "Select either architecture or train both.")], -1), W("span", null, A(I(Y).trainNanoWakeWordCnn && I(Y).trainNanoWakeWordGru ? "2 selected" : "1 selected"), 1)]), W("div", Dl, [W("button", {
							type: "button",
							class: k(["architecture-choice", { active: I(Y).trainNanoWakeWordCnn }]),
							"aria-pressed": I(Y).trainNanoWakeWordCnn,
							onClick: d[10] ||= (e) => I(Y).trainNanoWakeWordCnn = !I(Y).trainNanoWakeWordCnn
						}, [...d[84] ||= [
							W("i", null, "CNN", -1),
							W("span", null, [W("strong", null, "CNN"), W("small", null, "Recommended default with stronger recall.")], -1),
							W("b", null, "✓", -1)
						]], 10, Ol), W("button", {
							type: "button",
							class: k(["architecture-choice", { active: I(Y).trainNanoWakeWordGru }]),
							"aria-pressed": I(Y).trainNanoWakeWordGru,
							onClick: d[11] ||= (e) => I(Y).trainNanoWakeWordGru = !I(Y).trainNanoWakeWordGru
						}, [...d[85] ||= [
							W("i", null, "GRU", -1),
							W("span", null, [W("strong", null, "GRU"), W("small", null, "Recurrent model tuned for fewer false triggers.")], -1),
							W("b", null, "✓", -1)
						]], 10, kl)])])) : q("", !0),
						ee.value.length ? (H(), U("section", Al, [
							W("header", jl, [
								d[86] ||= W("div", { class: "hard-negative-icon" }, "−", -1),
								d[87] ||= W("div", null, [
									W("span", { class: "eyebrow" }, "Optional precision tuning"),
									W("strong", null, "Hard negative training"),
									W("small", null, "Teach the selected models which similar-sounding phrases must not wake the assistant.")
								], -1),
								W("b", null, A(se(I(Y).generatedNegativePhrases) ? `${se(I(Y).generatedNegativePhrases)} phrase${se(I(Y).generatedNegativePhrases) === 1 ? "" : "s"}` : "Optional"), 1)
							]),
							W("div", Ml, [d[88] ||= W("span", null, "Applied to", -1), (H(), U(V, null, R([
								"MWW",
								"OWW",
								"NWW"
							], (e) => W("i", {
								key: e,
								class: k({ active: ee.value.includes(e) })
							}, A(e), 3)), 64))]),
							W("label", Nl, [
								d[89] ||= W("span", null, "Phrases to reject", -1),
								L(W("textarea", {
									"onUpdate:modelValue": d[12] ||= (e) => I(Y).generatedNegativePhrases = e,
									rows: "4",
									placeholder: "hey later\nhey waiter\nokay tater"
								}, null, 512), [[Zo, I(Y).generatedNegativePhrases]]),
								d[90] ||= W("small", null, "One phrase per line. Each selected engine synthesizes and validates these near-misses alongside reviewed recordings and its broad negative data.", -1)
							]),
							d[91] ||= W("footer", null, [W("span", null, "Unchanged phrases reuse their cached audio on later runs."), W("strong", null, "The actual wake phrase is ignored automatically.")], -1)
						])) : q("", !0)
					]),
					!I(Y).training.running && I(Y).training.summary?.models?.length ? (H(), U("section", Pl, [W("header", null, [
						W("div", { class: k(["summary-mark", I(Y).training.summary.tone]) }, A(I(Y).training.summary.tone === "success" ? "✓" : I(Y).training.summary.tone === "error" ? "×" : "!"), 3),
						W("div", null, [
							d[92] ||= W("span", { class: "eyebrow" }, "Last completed run", -1),
							W("h4", null, A(I(Y).training.summary.wake_word), 1),
							W("p", null, [K(A(I(Y).training.summary.message), 1), I(Y).training.summary.completed_at ? (H(), U(V, { key: 0 }, [K(" · " + A(I(mc)(I(Y).training.summary.completed_at)), 1)], 64)) : q("", !0)])
						]),
						W("span", { class: k(["pill", I(Y).training.summary.tone]) }, A(I(Y).training.summary.overall), 3)
					]), W("div", Fl, [(H(!0), U(V, null, R(I(Y).training.summary.models, (e) => (H(), U("article", {
						key: e.id,
						class: k(e.tone)
					}, [
						W("header", null, [W("div", null, [W("span", null, A(e.id.replaceAll("_", " ")), 1), W("strong", null, A(e.label), 1)]), W("b", null, A(e.rating), 1)]),
						W("p", null, A(e.message), 1),
						e.metrics?.length ? (H(), U("div", Il, [(H(!0), U(V, null, R(e.metrics, (e) => (H(), U("span", { key: `${e.label}-${e.value}` }, [W("small", null, A(e.label), 1), W("strong", null, A(e.value), 1)]))), 128))])) : q("", !0)
					], 2))), 128))])])) : q("", !0),
					W("div", Ll, [
						W("button", {
							type: "button",
							class: "button primary large",
							disabled: !I(Y).session.safe_word || I(Y).training.running || I(X)("training-start"),
							onClick: d[13] ||= (...e) => I(cc) && I(cc)(...e)
						}, A(I(Y).training.running ? "Training in progress" : "Start training"), 9, Rl),
						W("button", {
							type: "button",
							class: "boundary-button",
							disabled: !x.value?.boundary_only_ready || I(Y).training.running || I(X)("boundary-start"),
							onClick: d[14] ||= (...e) => I(lc) && I(lc)(...e)
						}, A(I(X)("boundary-start") ? "Starting boundary…" : "Retrain boundary only"), 9, zl),
						b.value ? (H(), U("button", {
							key: 0,
							type: "button",
							disabled: I(Y).training.running || I(X)("nww-revalidate"),
							onClick: d[15] ||= (e) => I(uc)(String(b.value.id || "").replace("nano_", ""), String(I(Y).session.safe_word || ""))
						}, A(I(X)("nww-revalidate") ? "Checking NWW…" : `Revalidate ${String(b.value.label || "NWW")} candidate`), 9, Bl)) : q("", !0)
					]),
					W("footer", Vl, [d[93] ||= W("span", null, "Training opens the console automatically and continues if the window is closed.", -1), W("button", {
						type: "button",
						disabled: !I(As),
						onClick: d[16] ||= (e) => I(Y).consoleOpen = !0
					}, "Open console", 8, Hl)])
				])
			], 64)) : I(Y).activeView === "auto" ? (H(), U(V, { key: 1 }, [
				W("section", Ul, [d[95] ||= W("div", null, [
					W("span", { class: "eyebrow" }, "False-positive loop"),
					W("h2", null, "Auto Training"),
					W("p", null, "Transcribe captures, sort negatives, recover close misses, retrain on schedule, and publish through Tater.")
				], -1), W("span", { class: k(["pill hero-pill", m.value.tone]) }, A(m.value.text), 3)]),
				W("section", Wl, [
					d[105] ||= W("header", { class: "panel-head" }, [W("div", { class: "number" }, "1"), W("div", null, [W("h3", null, "Review rules"), W("p", null, "Conservative local STT keeps uncertain clips in the manual inbox.")])], -1),
					W("div", Gl, [
						W("label", null, [L(W("input", {
							"onUpdate:modelValue": d[17] ||= (e) => I(Y).autoForm.enabled = e,
							type: "checkbox"
						}, null, 512), [[Qo, I(Y).autoForm.enabled]]), d[96] ||= W("span", null, [W("strong", null, "Enable Auto Training"), W("small", null, "Queue eligible wake triggers for local transcription.")], -1)]),
						W("label", null, [L(W("input", {
							"onUpdate:modelValue": d[18] ||= (e) => I(Y).autoForm.delete_confirmed_wakes = e,
							type: "checkbox"
						}, null, 512), [[Qo, I(Y).autoForm.delete_confirmed_wakes]]), d[97] ||= W("span", null, [W("strong", null, "Delete confirmed good wakes"), W("small", null, "Remove normal triggers when STT confirms the phrase.")], -1)]),
						W("label", null, [L(W("input", {
							"onUpdate:modelValue": d[19] ||= (e) => I(Y).autoForm.promote_close_misses = e,
							type: "checkbox"
						}, null, 512), [[Qo, I(Y).autoForm.promote_close_misses]]), d[98] ||= W("span", null, [W("strong", null, "Promote confirmed close misses"), W("small", null, "Move verified close misses into positive samples.")], -1)])
					]),
					W("div", Kl, [
						W("label", ql, [d[99] ||= W("span", null, "Wake phrase", -1), L(W("input", {
							"onUpdate:modelValue": d[20] ||= (e) => I(Y).autoForm.wake_phrase = e,
							type: "text"
						}, null, 512), [[Zo, I(Y).autoForm.wake_phrase]])]),
						W("label", Jl, [d[100] ||= W("span", null, "STT language", -1), L(W("input", {
							"onUpdate:modelValue": d[21] ||= (e) => I(Y).autoForm.language = e,
							type: "text"
						}, null, 512), [[Zo, I(Y).autoForm.language]])]),
						String(I(Y).autoForm.language).toLowerCase().startsWith("en") ? (H(), U("label", Yl, [
							d[101] ||= W("span", null, "English accent emphasis", -1),
							L(W("select", { "onUpdate:modelValue": d[22] ||= (e) => I(Y).autoForm.english_accent = e }, [(H(!0), U(V, null, R(I(Y).englishAccents, (e) => (H(), U("option", {
								key: e.code,
								value: e.code
							}, A(e.label), 9, Xl))), 128))], 512), [[es, I(Y).autoForm.english_accent]]),
							d[102] ||= W("small", null, "Used when Auto Training needs to regenerate English TTS.", -1)
						])) : q("", !0),
						W("label", Zl, [
							d[103] ||= W("span", null, "STT engine", -1),
							L(W("select", { "onUpdate:modelValue": d[23] ||= (e) => I(Y).autoForm.stt_engine = e }, [(H(!0), U(V, null, R(I(Ns), (e) => (H(), U("option", {
								key: e.id || e.value,
								value: e.id || e.value
							}, A(e.label || e.name || e.id), 9, Ql))), 128))], 512), [[es, I(Y).autoForm.stt_engine]]),
							W("small", null, A(I(Ns).find((e) => (e.id || e.value) === I(Y).autoForm.stt_engine)?.description || "Runs locally on this trainer."), 1)
						]),
						W("label", $l, [d[104] ||= W("span", null, "Minimum transcript characters", -1), L(W("input", {
							"onUpdate:modelValue": d[24] ||= (e) => I(Y).autoForm.minimum_transcript_chars = e,
							min: "1",
							max: "100",
							type: "number"
						}, null, 512), [[
							Zo,
							I(Y).autoForm.minimum_transcript_chars,
							void 0,
							{ number: !0 }
						]])])
					])
				]),
				W("section", eu, [
					d[125] ||= W("header", { class: "panel-head" }, [W("div", { class: "number" }, "2"), W("div", null, [W("h3", null, "Training schedule"), W("p", null, "A run starts only after enough newly reviewed negatives accumulate.")])], -1),
					W("div", tu, [W("label", nu, [d[107] ||= W("span", null, "Run training", -1), L(W("select", { "onUpdate:modelValue": d[25] ||= (e) => I(Y).autoForm.schedule_hours = e }, [...d[106] ||= [
						W("option", { value: 0 }, "Manually only", -1),
						W("option", { value: 6 }, "Every 6 hours", -1),
						W("option", { value: 12 }, "Every 12 hours", -1),
						W("option", { value: 24 }, "Every day", -1),
						W("option", { value: 48 }, "Every 2 days", -1),
						W("option", { value: 168 }, "Every week", -1)
					]], 512), [[
						es,
						I(Y).autoForm.schedule_hours,
						void 0,
						{ number: !0 }
					]])]), W("label", ru, [d[108] ||= W("span", null, "Minimum new negatives", -1), L(W("input", {
						"onUpdate:modelValue": d[26] ||= (e) => I(Y).autoForm.minimum_new_negatives = e,
						min: "1",
						max: "10000",
						type: "number"
					}, null, 512), [[
						Zo,
						I(Y).autoForm.minimum_new_negatives,
						void 0,
						{ number: !0 }
					]])])]),
					W("div", iu, [
						W("div", au, [
							W("button", {
								type: "button",
								class: k(["model-choice", { active: I(Y).autoForm.train_microwakeword }]),
								"aria-pressed": I(Y).autoForm.train_microwakeword,
								onClick: d[27] ||= (e) => I(Y).autoForm.train_microwakeword = !I(Y).autoForm.train_microwakeword
							}, [...d[109] ||= [
								W("i", null, "MWW", -1),
								W("span", null, [W("strong", null, "MicroWakeWord"), W("small", null, "Refresh the ESP32 and ESPHome model.")], -1),
								W("b", null, "✓", -1)
							]], 10, ou),
							W("button", {
								type: "button",
								class: k(["model-choice", { active: I(Y).autoForm.train_openwakeword }]),
								"aria-pressed": I(Y).autoForm.train_openwakeword,
								onClick: d[28] ||= (e) => I(Y).autoForm.train_openwakeword = !I(Y).autoForm.train_openwakeword
							}, [...d[110] ||= [
								W("i", null, "OWW", -1),
								W("span", null, [W("strong", null, "openWakeWord"), W("small", null, "Refresh the openWakeWord ONNX model.")], -1),
								W("b", null, "✓", -1)
							]], 10, su),
							W("button", {
								type: "button",
								class: k(["model-choice", { active: T.value }]),
								"aria-pressed": T.value,
								onClick: O
							}, [...d[111] ||= [
								W("i", null, "NWW", -1),
								W("span", null, [W("strong", null, "NanoWakeWord"), W("small", null, "Refresh selected CNN and GRU models.")], -1),
								W("b", null, "✓", -1)
							]], 10, cu)
						]),
						T.value ? (H(), U("div", lu, [W("div", uu, [d[112] ||= W("div", null, [W("strong", null, "NanoWakeWord architectures"), W("small", null, "Select either architecture or retrain both.")], -1), W("span", null, A(I(Y).autoForm.train_nanowakeword_cnn && I(Y).autoForm.train_nanowakeword_gru ? "2 selected" : "1 selected"), 1)]), W("div", du, [W("button", {
							type: "button",
							class: k(["architecture-choice", { active: I(Y).autoForm.train_nanowakeword_cnn }]),
							"aria-pressed": I(Y).autoForm.train_nanowakeword_cnn,
							onClick: d[29] ||= (e) => I(Y).autoForm.train_nanowakeword_cnn = !I(Y).autoForm.train_nanowakeword_cnn
						}, [...d[113] ||= [
							W("i", null, "CNN", -1),
							W("span", null, [W("strong", null, "CNN"), W("small", null, "Recommended default with stronger recall.")], -1),
							W("b", null, "✓", -1)
						]], 10, fu), W("button", {
							type: "button",
							class: k(["architecture-choice", { active: I(Y).autoForm.train_nanowakeword_gru }]),
							"aria-pressed": I(Y).autoForm.train_nanowakeword_gru,
							onClick: d[30] ||= (e) => I(Y).autoForm.train_nanowakeword_gru = !I(Y).autoForm.train_nanowakeword_gru
						}, [...d[114] ||= [
							W("i", null, "GRU", -1),
							W("span", null, [W("strong", null, "GRU"), W("small", null, "Recurrent model tuned for fewer false triggers.")], -1),
							W("b", null, "✓", -1)
						]], 10, pu)])])) : q("", !0),
						te.value.length ? (H(), U("section", mu, [
							W("header", hu, [
								d[115] ||= W("div", { class: "hard-negative-icon" }, "−", -1),
								d[116] ||= W("div", null, [
									W("span", { class: "eyebrow" }, "Optional precision tuning"),
									W("strong", null, "Hard negative training"),
									W("small", null, "Keep near-miss protection consistent whenever Auto Training refreshes a model.")
								], -1),
								W("b", null, A(se(I(Y).autoForm.generated_negative_phrases) ? `${se(I(Y).autoForm.generated_negative_phrases)} phrase${se(I(Y).autoForm.generated_negative_phrases) === 1 ? "" : "s"}` : "Optional"), 1)
							]),
							W("div", gu, [d[117] ||= W("span", null, "Applied to", -1), (H(), U(V, null, R([
								"MWW",
								"OWW",
								"NWW"
							], (e) => W("i", {
								key: e,
								class: k({ active: te.value.includes(e) })
							}, A(e), 3)), 64))]),
							W("label", _u, [
								d[118] ||= W("span", null, "Phrases to reject", -1),
								L(W("textarea", {
									"onUpdate:modelValue": d[31] ||= (e) => I(Y).autoForm.generated_negative_phrases = e,
									rows: "4",
									placeholder: "hey later\nhey waiter\nokay tater"
								}, null, 512), [[Zo, I(Y).autoForm.generated_negative_phrases]]),
								d[119] ||= W("small", null, "One phrase per line. Saved with the schedule and synthesized for every selected model.", -1)
							]),
							d[120] ||= W("footer", null, [W("span", null, "Reviewed false wakes still receive the strongest model weight."), W("strong", null, "The actual wake phrase is ignored automatically.")], -1)
						])) : q("", !0)
					]),
					!I(Y).training.running && I(Y).training.summary?.models?.length ? (H(), U("section", vu, [W("header", null, [
						W("div", { class: k(["summary-mark", I(Y).training.summary.tone]) }, A(I(Y).training.summary.tone === "success" ? "✓" : I(Y).training.summary.tone === "error" ? "×" : "!"), 3),
						W("div", null, [
							d[121] ||= W("span", { class: "eyebrow" }, "Last completed run", -1),
							W("h4", null, A(I(Y).training.summary.wake_word), 1),
							W("p", null, [K(A(I(Y).training.summary.message), 1), I(Y).training.summary.completed_at ? (H(), U(V, { key: 0 }, [K(" · " + A(I(mc)(I(Y).training.summary.completed_at)), 1)], 64)) : q("", !0)])
						]),
						W("span", { class: k(["pill", I(Y).training.summary.tone]) }, A(I(Y).training.summary.overall), 3)
					]), W("div", yu, [(H(!0), U(V, null, R(I(Y).training.summary.models, (e) => (H(), U("article", {
						key: `auto-${e.id}`,
						class: k(e.tone)
					}, [
						W("header", null, [W("div", null, [W("span", null, A(e.id.replaceAll("_", " ")), 1), W("strong", null, A(e.label), 1)]), W("b", null, A(e.rating), 1)]),
						W("p", null, A(e.message), 1),
						e.metrics?.length ? (H(), U("div", bu, [(H(!0), U(V, null, R(e.metrics, (e) => (H(), U("span", { key: `${e.label}-${e.value}` }, [W("small", null, A(e.label), 1), W("strong", null, A(e.value), 1)]))), 128))])) : q("", !0)
					], 2))), 128))])])) : q("", !0),
					W("div", xu, [
						W("article", null, [d[122] ||= W("span", null, "Pending negatives", -1), W("strong", null, A(Number(u.value.pending_negative_count || 0)), 1)]),
						W("article", null, [d[123] ||= W("span", null, "Next check", -1), W("strong", Su, A(u.value.next_run_at ? I(mc)(u.value.next_run_at) : "Manual"), 1)]),
						W("article", null, [d[124] ||= W("span", null, "Last training", -1), W("strong", Cu, A(u.value.last_train_finished_at ? I(mc)(u.value.last_train_finished_at) : "Never"), 1)])
					])
				]),
				W("section", wu, [
					d[129] ||= W("header", { class: "panel-head" }, [W("div", { class: "number" }, "3"), W("div", null, [W("h3", null, "Publish to Tater"), W("p", null, "Securely activate successful models across every connected satellite.")])], -1),
					W("div", Tu, [W("label", Eu, [
						d[126] ||= W("span", null, "Trainer public URL", -1),
						L(W("input", {
							"onUpdate:modelValue": d[32] ||= (e) => I(Y).autoForm.advertised_base_url = e,
							type: "text",
							placeholder: "Auto-detect LAN address"
						}, null, 512), [[Zo, I(Y).autoForm.advertised_base_url]]),
						W("small", null, A(I(Y).autoForm.advertised_base_url ? `Configured: ${I(Y).autoForm.advertised_base_url}` : `Detected: ${I(Y).auto.advertised_base_url || "unavailable"}`), 1)
					]), W("label", Du, [d[127] ||= W("span", null, "Tater URL", -1), L(W("input", {
						"onUpdate:modelValue": d[33] ||= (e) => I(Y).autoForm.tater_url = e,
						type: "text"
					}, null, 512), [[Zo, I(Y).autoForm.tater_url]])])]),
					W("div", Ou, [
						W("span", { class: k(["pill", I(Ms) ? "success" : "warning"]) }, A(I(Ms) ? `Linked${I(Y).auto.trainer_link?.tater_name ? ` · ${I(Y).auto.trainer_link.tater_name}` : ""}` : "Not linked"), 3),
						W("button", {
							type: "button",
							class: "button primary",
							disabled: I(X)("auto"),
							onClick: ce
						}, A(I(Ms) ? "Relink Tater" : "Link Tater"), 9, ku),
						I(Ms) ? (H(), U("button", {
							key: 0,
							type: "button",
							class: "button danger",
							disabled: I(X)("auto"),
							onClick: d[34] ||= (...e) => I(tc) && I(tc)(...e)
						}, "Unlink", 8, Au)) : q("", !0)
					]),
					W("div", ju, [W("label", null, [L(W("input", {
						"onUpdate:modelValue": d[35] ||= (e) => I(Y).autoForm.notify_satellites = e,
						type: "checkbox"
					}, null, 512), [[Qo, I(Y).autoForm.notify_satellites]]), d[128] ||= W("span", null, [W("strong", null, "Activate after successful training"), W("small", null, "Tater applies the new word globally.")], -1)])])
				]),
				W("section", Mu, [W("div", Nu, [
					W("button", {
						type: "button",
						class: "button primary",
						disabled: I(X)("auto"),
						onClick: d[36] ||= (...e) => I(Qs) && I(Qs)(...e)
					}, "Save Auto Training", 8, Pu),
					W("button", {
						type: "button",
						disabled: I(X)("auto"),
						onClick: d[37] ||= (e) => I($s)("review_now")
					}, "Review inbox now", 8, Fu),
					W("button", {
						type: "button",
						disabled: I(X)("auto") || I(Y).training.running,
						onClick: d[38] ||= (e) => I($s)("train_now")
					}, "Train now", 8, Iu),
					W("button", {
						type: "button",
						disabled: I(X)("auto") || !I(Ms),
						onClick: d[39] ||= (e) => I($s)("notify_now")
					}, "Publish current word", 8, Lu)
				]), W("p", Ru, A(f.value), 1)])
			], 64)) : I(Y).activeView === "captured" ? (H(), U(V, { key: 2 }, [
				W("section", zu, [d[130] ||= W("div", null, [
					W("span", { class: "eyebrow" }, "Capture review"),
					W("h2", null, "Captured Audio"),
					W("p", null, "Listen to clips from your satellites and turn every real-world event into a better model.")
				], -1), W("span", { class: k(["pill hero-pill", I(Y).captured.captured_count ? "warning" : ""]) }, A(I(Y).captured.captured_count ? `${I(Y).captured.captured_count} waiting` : "Inbox idle"), 3)]),
				W("section", Bu, [W("header", Vu, [
					d[131] ||= W("div", { class: "number" }, "1", -1),
					d[132] ||= W("div", null, [W("h3", null, "Review queue"), W("p", null, "Approve good phrases, keep false positives as negatives, or discard noise.")], -1),
					W("button", {
						type: "button",
						disabled: I(X)("captured"),
						onClick: d[40] ||= (e) => I(Hs)()
					}, A(I(X)("captured") ? "Refreshing…" : "Refresh inbox"), 9, Hu)
				]), W("div", Uu, [
					W("article", null, [d[133] ||= W("span", null, "Inbox", -1), W("strong", null, A(I(Y).captured.captured_count), 1)]),
					W("article", null, [d[134] ||= W("span", null, "All-word negatives", -1), W("strong", null, A(I(Y).captured.negative_count), 1)]),
					W("article", null, [d[135] ||= W("span", null, "All-word positives", -1), W("strong", null, A(I(Y).captured.personal_count), 1)])
				])]),
				W("section", Wu, [d[140] ||= W("header", { class: "panel-head" }, [W("div", { class: "number" }, "2"), W("div", null, [W("h3", null, "Listen + sort"), W("p", null, "Metadata remains visible so borderline detections are easy to understand.")])], -1), I(Y).captured.items?.length ? (H(), U("div", Ku, [(H(!0), U(V, null, R(I(Y).captured.items, (e) => (H(), U("article", {
					key: e.capture_id || e.saved_as,
					class: "audio-card"
				}, [
					W("header", null, [W("div", null, [W("strong", null, A(e.original_name || e.saved_as), 1), W("small", null, A(I(mc)(e.captured_at || e.received_at)) + " " + A(e.message || ""), 1)]), W("span", { class: k(["pill", I(_c)(e).tone]) }, A(I(_c)(e).label), 3)]),
					fe(e).length ? (H(), U("div", qu, [(H(!0), U(V, null, R(fe(e), (e) => (H(), U("span", { key: e }, A(e), 1))), 128))])) : q("", !0),
					W("div", { class: k(["capture-assignment", { unassigned: !e.review_wake_word }]) }, [W("label", null, [d[137] ||= W("span", null, "Train this clip for", -1), L(W("select", { "onUpdate:modelValue": (t) => e.review_wake_word = t }, [d[136] ||= W("option", { value: "" }, "Choose a saved wake word…", -1), (H(!0), U(V, null, R(I(Y).wakeWordLibrary, (e) => (H(), U("option", {
						key: e.safe_word,
						value: e.safe_word
					}, A(e.phrase) + " · " + A(e.safe_word), 9, Yu))), 128))], 8, Ju), [[es, e.review_wake_word]])]), W("span", { class: k(["pill", e.review_wake_word ? "success" : "warning"]) }, A(e.review_wake_word ? `Routed to ${e.review_wake_word}` : "Unassigned"), 3)], 2),
					e.transcript ? (H(), U("div", Xu, [d[138] ||= W("b", null, "STT", -1), K(" " + A(e.transcript), 1)])) : q("", !0),
					e.auto_review_guided_transcript ? (H(), U("div", Zu, [d[139] ||= W("b", null, "Guided wake check", -1), K(" " + A(e.auto_review_guided_transcript), 1)])) : q("", !0),
					W("audio", {
						controls: "",
						preload: "none",
						src: I(vc)(e, "captured")
					}, null, 8, Qu),
					W("footer", null, [W("span", null, A(e.saved_as) + " · " + A(I(gc)(e.final_format)), 1), W("div", null, [
						W("button", {
							type: "button",
							disabled: I(X)("review") || !e.review_wake_word,
							onClick: (t) => I(Ks)(e, "approve_personal")
						}, "Add positive", 8, $u),
						W("button", {
							type: "button",
							disabled: I(X)("review") || !e.review_wake_word,
							onClick: (t) => I(Ks)(e, "mark_negative")
						}, "Mark negative", 8, ed),
						W("button", {
							type: "button",
							class: "button danger ghost",
							disabled: I(X)("review"),
							onClick: (t) => I(Ks)(e, "discard")
						}, "Discard", 8, td)
					])])
				]))), 128))])) : (H(), U("div", Gu, "No captured audio yet. Clips sent by satellites will appear here."))])
			], 64)) : I(Y).activeView === "samples" ? (H(), U(V, { key: 3 }, [
				W("section", nd, [d[141] ||= W("div", null, [
					W("span", { class: "eyebrow" }, "Sample library"),
					W("h2", null, "Current Training Samples"),
					W("p", null, "Audit positives and negatives, trim recordings precisely, and import seed audio.")
				], -1), W("span", rd, A(I(Es) + I(Ds)) + " total", 1)]),
				W("section", id, [
					W("header", ad, [
						d[144] ||= W("div", { class: "number" }, "1", -1),
						d[145] ||= W("div", null, [W("h3", null, "Saved samples"), W("p", null, "Personal clips are positives. Negative clips are false wakes and hard negatives.")], -1),
						W("div", od, [W("button", {
							type: "button",
							class: k({ active: I(Y).sampleBucket === "personal" }),
							onClick: d[41] ||= (e) => ie("personal")
						}, [d[142] ||= K("Personal ", -1), W("b", null, A(I(Es)), 1)], 2), W("button", {
							type: "button",
							class: k({ active: I(Y).sampleBucket === "negative" }),
							onClick: d[42] ||= (e) => ie("negative")
						}, [d[143] ||= K("Negative ", -1), W("b", null, A(I(Ds)), 1)], 2)])
					]),
					W("div", sd, [
						W("button", {
							type: "button",
							disabled: I(X)("samples"),
							onClick: d[43] ||= (e) => I(Vs)()
						}, "Refresh", 8, cd),
						W("button", {
							type: "button",
							class: "button danger ghost",
							disabled: I(X)("review") || I(Es) === 0,
							onClick: d[44] ||= (e) => I(Ys)("personal")
						}, "Clear positives", 8, ld),
						W("button", {
							type: "button",
							class: "button danger ghost",
							disabled: I(X)("review") || I(Ds) === 0,
							onClick: d[45] ||= (e) => I(Ys)("negative")
						}, "Clear negatives", 8, ud)
					]),
					I(js).length ? (H(), U("div", fd, [(H(!0), U(V, null, R(c.value, (e) => (H(), U("article", {
						key: e.saved_as,
						class: "audio-card"
					}, [
						W("header", null, [W("div", null, [W("strong", null, A(e.saved_as), 1), W("small", null, A(pe(e)), 1)]), W("div", pd, [e.trimmed ? (H(), U("span", md, "Trimmed")) : q("", !0), W("span", { class: k(["pill", I(Y).sampleBucket === "personal" ? "success" : "error"]) }, A(I(Y).sampleBucket === "personal" ? "Positive" : "Negative"), 3)])]),
						e.transcript ? (H(), U("div", hd, [d[146] ||= W("b", null, "STT", -1), K(" " + A(e.transcript), 1)])) : q("", !0),
						e.auto_review_guided_transcript ? (H(), U("div", gd, [d[147] ||= W("b", null, "Guided wake check", -1), K(" " + A(e.auto_review_guided_transcript), 1)])) : q("", !0),
						W("audio", {
							controls: "",
							preload: "none",
							src: I(vc)(e, I(Y).sampleBucket)
						}, null, 8, _d),
						W("footer", null, [W("span", null, A(I(gc)(e.final_format)), 1), W("div", null, [
							W("button", {
								type: "button",
								onClick: (t) => ae(e, I(Y).sampleBucket)
							}, "Trim", 8, vd),
							e.trimmed ? (H(), U("button", {
								key: 0,
								type: "button",
								onClick: (t) => I(Js)(e, I(Y).sampleBucket)
							}, "Revert", 8, yd)) : q("", !0),
							W("button", {
								type: "button",
								class: "button danger ghost",
								disabled: I(X)("review"),
								onClick: (t) => I(qs)(e, I(Y).sampleBucket)
							}, "Remove", 8, bd)
						])])
					]))), 128))])) : (H(), U("div", dd, "No " + A(I(Y).sampleBucket) + " samples saved yet.", 1)),
					l.value > 1 ? (H(), U("div", xd, [
						W("button", {
							type: "button",
							disabled: I(Y).samplePage[I(Y).sampleBucket] === 0,
							onClick: d[46] ||= (e) => I(Y).samplePage[I(Y).sampleBucket]--
						}, "Previous", 8, Sd),
						W("span", null, "Page " + A(I(Y).samplePage[I(Y).sampleBucket] + 1) + " of " + A(l.value), 1),
						W("button", {
							type: "button",
							disabled: I(Y).samplePage[I(Y).sampleBucket] >= l.value - 1,
							onClick: d[47] ||= (e) => I(Y).samplePage[I(Y).sampleBucket]++
						}, "Next", 8, Cd)
					])) : q("", !0)
				]),
				W("section", wd, [
					d[149] ||= W("header", { class: "panel-head" }, [W("div", { class: "number" }, "2"), W("div", null, [W("h3", null, "Manual sample import"), W("p", null, "Optional seed recordings are normalized to the trainer’s required WAV format.")])], -1),
					W("label", Td, [
						W("input", {
							ref_key: "uploadInput",
							ref: t,
							type: "file",
							multiple: "",
							accept: "audio/*,.wav,.mp3,.m4a,.flac,.ogg,.aac,.webm,.opus",
							onChange: d[48] ||= (...e) => I(Us) && I(Us)(...e)
						}, null, 544),
						d[148] ||= W("span", null, [W("strong", null, "Choose one or many audio files"), W("small", null, "WAV, MP3, M4A, FLAC, OGG, AAC, OPUS, and WEBM")], -1),
						W("b", null, A(I(Y).selectedFiles.length ? `${I(Y).selectedFiles.length} selected` : "Browse"), 1)
					]),
					W("button", {
						type: "button",
						class: "button primary",
						disabled: !I(Y).session.safe_word || !I(Y).selectedFiles.length || I(X)("upload"),
						onClick: d[49] ||= (e) => I(Gs)(t.value)
					}, A(I(X)("upload") ? "Uploading…" : "Upload selected samples"), 9, Ed),
					W("div", Dd, [
						W("div", null, [W("strong", null, A(I(Y).uploadLabel), 1), W("span", null, A(I(Y).uploadProgress) + "%", 1)]),
						W("div", Od, [W("i", { style: de({ width: `${I(Y).uploadProgress}%` }) }, null, 4)]),
						W("small", null, A(I(Y).uploadDetail), 1)
					])
				])
			], 64)) : I(Y).activeView === "data" ? (H(), U(V, { key: 4 }, [
				W("section", kd, [d[150] ||= W("div", null, [
					W("span", { class: "eyebrow" }, "Local storage"),
					W("h2", null, "Data Management"),
					W("p", null, "See exactly what the trainer has downloaded, generated, recorded, and produced.")
				], -1), W("span", Ad, A(I(hc)(I(Y).managedData.total_size_bytes)) + " total", 1)]),
				W("section", jd, [
					W("header", Md, [
						d[151] ||= W("div", { class: "number" }, "i", -1),
						d[152] ||= W("div", null, [W("h3", null, "Trainer storage"), W("p", null, "Deleting an item is permanent. Required downloads and generated caches will be rebuilt the next time training needs them.")], -1),
						W("button", {
							type: "button",
							disabled: I(X)("data") || I(X)("data-delete"),
							onClick: d[50] ||= (e) => I(ac)()
						}, A(I(X)("data") ? "Scanning…" : "Refresh sizes"), 9, Nd)
					]),
					W("div", Pd, [
						W("article", null, [d[153] ||= W("span", null, "Space used", -1), W("strong", Fd, A(I(hc)(I(Y).managedData.total_size_bytes)), 1)]),
						W("article", null, [d[154] ||= W("span", null, "Files", -1), W("strong", null, A(Number(I(Y).managedData.total_file_count || 0).toLocaleString()), 1)]),
						W("article", null, [d[155] ||= W("span", null, "Individual items", -1), W("strong", null, A(I(Y).managedData.items.length), 1)])
					]),
					I(Y).training.running ? (H(), U("p", Id, "Stop the active training session before deleting data.")) : q("", !0)
				]),
				(H(!0), U(V, null, R(C.value, (e, t) => (H(), U("section", {
					key: e.name,
					class: "panel data-panel"
				}, [W("header", Ld, [W("div", Rd, A(t + 1), 1), W("div", null, [W("h3", null, A(e.name), 1), W("p", null, A(e.items.length) + " separately managed item" + A(e.items.length === 1 ? "" : "s"), 1)])]), W("div", zd, [(H(!0), U(V, null, R(e.items, (e) => (H(), U("article", {
					key: e.id,
					class: k(["data-row", { empty: !e.file_count }])
				}, [
					W("div", Bd, [
						W("div", Vd, [W("strong", null, A(e.label), 1), W("code", null, A(e.location), 1)]),
						W("small", null, A(e.description), 1),
						e.rebuild_note ? (H(), U("span", Hd, A(e.rebuild_note), 1)) : q("", !0)
					]),
					W("div", Ud, [W("strong", null, A(I(hc)(e.size_bytes)), 1), W("span", null, A(Number(e.file_count || 0).toLocaleString()) + " file" + A(e.file_count === 1 ? "" : "s"), 1)]),
					W("button", {
						type: "button",
						class: "button danger ghost",
						disabled: !e.file_count || I(Y).training.running || I(X)("data") || I(X)("data-delete"),
						onClick: (t) => I(oc)(e)
					}, A(I(X)("data-delete") ? "Please wait…" : "Delete"), 9, Wd)
				], 2))), 128))])]))), 128)),
				!I(X)("data") && !I(Y).managedData.items.length ? (H(), U("section", Gd, "No managed trainer data was found.")) : q("", !0)
			], 64)) : I(Y).activeView === "firmware" ? (H(), U(V, { key: 5 }, [
				W("section", Kd, [d[156] ||= W("div", null, [
					W("span", { class: "eyebrow" }, "Wake-word catalog"),
					W("h2", null, "Trained Wake Words"),
					W("p", null, "Copy a local JSON package URL into Tater to switch every native satellite live.")
				], -1), W("span", { class: k(["pill hero-pill", I(Y).wakeWords.length ? "success" : "warning"]) }, A(I(Y).wakeWords.length ? `${I(Y).wakeWords.length} trained` : "Catalog empty"), 3)]),
				d[163] ||= W("div", { class: "native-notice" }, [W("strong", null, "Tater Native"), W("span", null, "These packages include model metadata and a direct model URL for live satellite updates.")], -1),
				W("section", qd, [W("header", Jd, [
					d[157] ||= W("div", { class: "number" }, "v1", -1),
					d[158] ||= W("div", null, [W("h3", null, "Published model URLs"), W("p", null, "URLs stay local and are refreshed after each successful run.")], -1),
					W("button", {
						type: "button",
						disabled: I(X)("firmware") || I(X)("library"),
						onClick: d[51] ||= (e) => {
							I(nc)(), I(rc)();
						}
					}, "Refresh", 8, Yd)
				]), I(Y).wakeWords.length ? (H(), U("div", Zd, [(H(!0), U(V, null, R(I(Y).wakeWords, (e) => (H(), U("article", { key: e.key || me(e) }, [W("div", null, [
					W("strong", null, A(e.label || e.name || "Trained wake word"), 1),
					me(e) ? (H(), U("a", {
						key: 0,
						href: me(e),
						target: "_blank",
						rel: "noreferrer"
					}, "Tater wake-word bundle · " + A(me(e)), 9, Qd)) : (H(), U("span", $d, "Bundle URL unavailable")),
					ge(e) ? (H(), U("a", {
						key: 2,
						href: ge(e),
						target: "_blank",
						rel: "noreferrer"
					}, "MicroWakeWord model · " + A(ge(e)), 9, ef)) : q("", !0),
					_e(e) ? (H(), U("a", {
						key: 3,
						href: _e(e),
						target: "_blank",
						rel: "noreferrer"
					}, "openWakeWord model · " + A(_e(e)), 9, tf)) : q("", !0),
					(H(!0), U(V, null, R(ye(e), (e, t) => (H(), U("a", {
						key: t,
						href: e,
						target: "_blank",
						rel: "noreferrer"
					}, "NanoWakeWord " + A(String(t).toUpperCase()) + " model · " + A(e), 9, nf))), 128)),
					ve(e) ? (H(), U("a", {
						key: 4,
						href: ve(e),
						target: "_blank",
						rel: "noreferrer"
					}, "Wake boundary model · " + A(ve(e)), 9, rf)) : q("", !0),
					W("div", af, [
						(H(!0), U(V, null, R(e.available_engines || [], (e) => (H(), U("span", { key: e }, A(String(e).replaceAll("_", " ")), 1))), 128)),
						e.wake_boundary_available ? (H(), U("span", of, "wake boundary · threshold " + A(e.wake_boundary_threshold), 1)) : q("", !0),
						e.language ? (H(), U("span", sf, A(e.language), 1)) : q("", !0),
						e.trained_at ? (H(), U("span", cf, A(I(mc)(e.trained_at)), 1)) : q("", !0),
						e.recall === void 0 ? q("", !0) : (H(), U("span", lf, "recall " + A(e.recall), 1))
					])
				]), W("button", {
					type: "button",
					disabled: !me(e),
					onClick: (t) => I(sc)(me(e))
				}, "Copy bundle URL", 8, uf)]))), 128))])) : (H(), U("div", Xd, "Train a wake word and its package will appear here."))]),
				W("section", df, [W("header", ff, [
					d[159] ||= W("div", { class: "number" }, "↗", -1),
					d[160] ||= W("div", null, [W("h3", null, "Model history + quality"), W("p", null, "Immutable snapshots let you compare each word’s detector and boundary results over time.")], -1),
					W("span", { class: k(["pill", S.value.length ? "success" : ""]) }, A(I(Y).modelHistory.length) + " runs", 3)
				]), S.value.length ? (H(), U("div", pf, [(H(!0), U(V, null, R(S.value, (e) => (H(), U("article", { key: e.run_id }, [W("div", mf, [W("i", { class: k(e.summary?.tone || "neutral") }, null, 2)]), W("div", hf, [
					W("header", null, [W("div", null, [W("span", null, A(e.run_kind === "boundary" ? "Boundary only" : "Model training"), 1), W("strong", null, A(e.wake_word), 1)]), W("b", { class: k(["pill", e.summary?.tone || ""]) }, A(e.summary?.overall || "Completed"), 3)]),
					W("p", null, A(I(mc)(e.completed_at)) + " · " + A((e.selected_models || []).map((e) => e.replaceAll("_", " ")).join(" + ")), 1),
					W("div", gf, [(H(!0), U(V, null, R(e.summary?.models || [], (e) => (H(), U("div", {
						key: e.id,
						class: k(e.tone)
					}, [W("span", null, [W("strong", null, A(e.label), 1), W("b", null, A(e.rating), 1)]), (H(!0), U(V, null, R(e.metrics || [], (e) => (H(), U("small", { key: `${e.label}-${e.value}` }, [K(A(e.label) + " ", 1), W("b", null, A(e.value), 1)]))), 128))], 2))), 128))])
				])]))), 128))])) : (H(), U("div", _f, "Completed training runs will appear here with their quality metrics."))]),
				d[164] ||= W("div", { class: "native-notice esphome-notice" }, [W("strong", null, "ESPHome"), W("span", null, "Strict micro_wake_word manifest without Tater Native or calibration extensions.")], -1),
				W("section", vf, [d[162] ||= W("header", { class: "panel-head" }, [W("div", { class: "number" }, "ESP"), W("div", null, [W("h3", null, "ESPHome JSON"), W("p", null, "Use this URL as the model in an ESPHome micro_wake_word configuration.")])], -1), I(Y).wakeWords.length ? (H(), U("div", bf, [(H(!0), U(V, null, R(I(Y).wakeWords, (e) => (H(), U("article", { key: `esphome-${e.key || he(e)}` }, [W("div", null, [
					W("strong", null, A(e.label || e.name || "Trained wake word"), 1),
					he(e) ? (H(), U("a", {
						key: 0,
						href: he(e),
						target: "_blank",
						rel: "noreferrer"
					}, "ESPHome JSON · " + A(he(e)), 9, xf)) : (H(), U("span", Sf, "ESPHome package URL unavailable")),
					d[161] ||= W("div", { class: "meta-row" }, [W("span", null, "Schema v2"), W("span", null, "Same TFLite model")], -1)
				]), W("button", {
					type: "button",
					disabled: !he(e),
					onClick: (t) => I(sc)(he(e))
				}, "Copy ESPHome URL", 8, Cf)]))), 128))])) : (H(), U("div", yf, "ESPHome links appear after a wake word is trained."))])
			], 64)) : q("", !0)], 64)) : (H(), U("div", Vc, [...d[60] ||= [W("span", { class: "spinner" }, null, -1), W("strong", null, "Connecting to the local trainer…", -1)]]))]),
			(H(), ia(Yn, { to: "body" }, [I(Y).consoleOpen ? (H(), U("div", {
				key: 0,
				class: "modal-backdrop console-backdrop",
				onClick: d[53] ||= os((e) => I(Y).consoleOpen = !1, ["self"])
			}, [W("section", wf, [
				W("header", Tf, [d[165] ||= W("div", null, [
					W("span", { class: "eyebrow" }, "Live pipeline"),
					W("h2", null, "Training Console"),
					W("p", null, "Closing this window does not interrupt training.")
				], -1), W("div", Ef, [
					r.value ? q("", !0) : (H(), U("button", {
						key: 0,
						type: "button",
						class: "console-follow",
						onClick: D
					}, "Jump to latest")),
					W("span", { class: k(["pill", p.value.tone]) }, A(p.value.text), 3),
					W("button", {
						type: "button",
						onClick: d[52] ||= (e) => I(Y).consoleOpen = !1
					}, "Close")
				])]),
				g.value ? (H(), U("div", {
					key: 0,
					class: k(["console-progress", {
						indeterminate: v.value === null && !g.value.status,
						complete: g.value.status === "complete",
						failed: g.value.status === "failed"
					}])
				}, [
					W("div", Df, [W("div", null, [W("span", null, A(_.value), 1), W("strong", null, A(g.value.label || "Working…"), 1)]), W("b", null, A(g.value.status === "failed" ? "Failed" : g.value.status === "complete" ? "Done" : v.value === null ? "Working" : `${v.value.toFixed(1)}%`), 1)]),
					W("div", Of, [W("i", { style: de({ width: g.value.status ? "100%" : v.value === null ? "38%" : `${v.value}%` }) }, null, 4)]),
					y.value.length ? (H(), U("div", kf, [(H(!0), U(V, null, R(y.value, (e) => (H(), U("span", { key: e }, A(e), 1))), 128))])) : q("", !0)
				], 2)) : q("", !0),
				W("pre", {
					ref_key: "consoleLog",
					ref: n,
					class: "console-log",
					onScrollPassive: ne
				}, [(H(!0), U(V, null, R(h.value, (e, t) => (H(), U("span", {
					key: `${t}-${e}`,
					class: k(be(e))
				}, A(e), 3))), 128))], 544)
			])])) : q("", !0)])),
			(H(), ia(Yn, { to: "body" }, [I(Y).taterLinkOpen ? (H(), U("div", {
				key: 0,
				class: "modal-backdrop",
				onClick: d[57] ||= os((e) => I(Y).taterLinkOpen = !1, ["self"])
			}, [W("section", Af, [W("header", jf, [W("div", null, [
				d[166] ||= W("span", { class: "eyebrow" }, "Secure pairing", -1),
				W("h2", null, A(o.value ? "Tater linked" : "Link Tater"), 1),
				W("p", null, A(o.value ? "This trainer can securely publish wake-word updates." : "Enter the short-lived code shown in Tater Voice Settings."), 1)
			]), W("button", {
				type: "button",
				onClick: d[54] ||= (e) => I(Y).taterLinkOpen = !1
			}, "Close")]), o.value ? (H(), U("div", Mf, [
				d[167] ||= W("i", null, "✓", -1),
				W("strong", null, "Successfully linked" + A(I(Y).auto.trainer_link?.tater_name ? ` to ${I(Y).auto.trainer_link.tater_name}` : ""), 1),
				d[168] ||= W("span", null, "The private link key is stored locally and is never displayed.", -1)
			])) : (H(), U("div", Nf, [
				W("label", Pf, [d[169] ||= W("span", null, "Tater address", -1), L(W("input", {
					"onUpdate:modelValue": d[55] ||= (e) => i.value = e,
					type: "text"
				}, null, 512), [[Zo, i.value]])]),
				W("label", Ff, [d[170] ||= W("span", null, "Tater pairing code", -1), L(W("input", {
					id: "pairing-code",
					"onUpdate:modelValue": d[56] ||= (e) => a.value = e,
					class: "pairing-code",
					maxlength: "9",
					placeholder: "ABCD-EFGH",
					autocomplete: "off",
					onInput: le
				}, null, 544), [[Zo, a.value]])]),
				d[171] ||= W("small", null, "In Tater, open Voice Settings → Wake Word Trainer → Link Trainer.", -1),
				W("button", {
					type: "button",
					class: "button primary",
					disabled: I(X)("link"),
					onClick: ue
				}, A(I(X)("link") ? "Linking securely…" : "Link Tater"), 9, If)
			]))])])) : q("", !0)])),
			G(Ac),
			G(eo, { name: "toast" }, {
				default: On(() => [I(Y).toast.message ? (H(), U("div", {
					key: 0,
					class: k(["toast", I(Y).toast.tone]),
					role: "status"
				}, A(I(Y).toast.message), 3)) : q("", !0)]),
				_: 1
			})
		]));
	}
}), Bf = document.getElementById("trainer-app");
if (!Bf) throw Error("Missing #trainer-app mount point");
fs(zf).mount(Bf);
//#endregion
